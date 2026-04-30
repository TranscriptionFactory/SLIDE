import numpy as np
import pandas as pd
import os
import datetime
from glob import glob
import pickle
from tqdm import tqdm
from itertools import product
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO

from .tools import init_data, calc_default_fsize, show_params
from .love import call_love
from .knockoffs import Knockoffs

from .plotting import Plotter
from .score import Estimator, SLIDE_Estimator


class SLIDE:
    def __init__(self, input_params, x=None, y=None):
        self.data, self.input_params = init_data(input_params, x, y)

    def calc_default_fsize(self, K):
        """
        Calculate the default feature size for the SLIDE algorithm.
        This breaks the data into many chunks to run knockoffs on each separately.

        Args:
            K (int): Number of latent factors.
        """
        n_rows = self.data.X.shape[0]
        return self.input_params.get("f_size", calc_default_fsize(n_rows, K))

    def show_params(self):
        """
        Display the parameters and data.
        """
        show_params(self.input_params, self.data)

    def load_love(self, love_res_path):
        """
        Load the LOVE result and calculate the latent factors.
        This is used when we want to continue running SLIDE from a previous LOVE run.
        """
        try:
            with open(love_res_path, "rb") as f:
                love_result = pickle.load(f)

            self.A = pd.DataFrame(
                love_result["A"],
                index=self.data.X.columns,
                columns=[f"Z{i}" for i in range(love_result["A"].shape[1])],
            )

            self.latent_factors = self.calc_z_matrix(love_result)
            self.love_result = love_result

        except Exception as e:
            print(f"Error loading LOVE result: {e}")
            return

    def load_state(self, out_iter):
        try:
            self.A = pd.read_csv(os.path.join(out_iter, "A.csv"), index_col=0)
            self.latent_factors = pd.read_csv(
                os.path.join(out_iter, "z_matrix.csv"), index_col=0
            )

            interact_path = os.path.join(out_iter, "sig_interacts.txt")
            if os.path.exists(interact_path):
                self.sig_interacts = (
                    np.loadtxt(interact_path, dtype=str).reshape(-1).tolist()
                )
            else:
                self.sig_interacts = []

            self.sig_LFs = (
                np.loadtxt(os.path.join(out_iter, "sig_LFs.txt"), dtype=str)
                .reshape(-1)
                .tolist()
            )
            self.marginal_idxs = np.where(
                self.latent_factors.columns.isin(self.sig_LFs)
            )[0]

        except Exception as e:
            print(f"No previous state found for {out_iter}")
            self.marginal_idxs = []

    @staticmethod
    def get_LF_genes(A, lf, X, y, lf_thresh=0.05, top_feats=20, outpath=None):
        """
        Select the top features associated with a latent factor.

        Matches the R SLIDE approach: top N/2 by absolute loading, then
        top N/2 by absolute correlation from the remaining features.

        Parameters:
        - A: Loading matrix (features x latent factors).
        - lf: Column name in A for the latent factor.
        - X: Feature data matrix.
        - y: Response variable.
        - top_feats: Total number of features to return (split between loading and correlation).
        - outpath: Optional directory to save per-LF feature list.

        Returns:
        - DataFrame with 'loading', 'AUC', 'corr', 'color' columns for selected features.
        """

        if lf not in A.columns:
            raise ValueError(f"Latent factor {lf} not found in A matrix")

        # Get ALL non-zero features (no threshold — matches R behavior)
        all_genes = A.loc[A[lf].abs() > 0, lf]

        # Compute correlation for all candidates (cheap)
        lf_info = pd.DataFrame(
            index=all_genes.index, columns=["loading", "AUC", "corr", "color"]
        )
        lf_info["loading"] = all_genes
        lf_info["corr"] = [
            np.corrcoef(X[x].values.flatten(), y.values.flatten())[0, 1]
            for x in all_genes.index
        ]

        # Select top N/2 by absolute loading
        top_loading = lf_info.sort_values(by="loading", key=abs, ascending=False).head(
            top_feats // 2
        )

        # Select top N/2 by absolute correlation from remaining features
        remaining = lf_info.drop(top_loading.index)
        top_corr = remaining.sort_values(by="corr", key=abs, ascending=False).head(
            top_feats // 2
        )

        # Tag groups for plot ordering (loading at bottom, corr on top)
        top_loading = top_loading.copy()
        top_corr = top_corr.copy()
        top_loading["group"] = "loading"
        top_corr["group"] = "corr"

        # Combine and compute AUC only for selected features (expensive)
        selected = pd.concat([top_loading, top_corr], axis=0)
        scorer = Estimator(model="auto", scaler="standard")
        selected["AUC"] = np.array(
            [scorer.evaluate(X[x], y, n_iters=3) for x in selected.index]
        ).mean(axis=1)

        # Color: red = positive corr, blue = negative corr (matches R)
        selected["color"] = np.where(selected["corr"] > 0, "red", "blue")

        selected = selected.sort_values(by="loading", key=abs, ascending=False)

        if outpath is not None:
            selected.to_csv(os.path.join(outpath, f"feature_list_{lf}.csv"), sep="\t")

        return selected

    def save_params(self, outpath, scores):
        """
        Save the parameters and scores.
        """
        if scores is None:
            true_scores = "NA"
            partial_random = "NA"
            full_random = "NA"
        else:
            true_scores = scores["real"]
            partial_random = np.mean(
                [x for x in scores["partial_random"] if x is not None]
            )
            full_random = np.mean([x for x in scores["full_random"] if x is not None])

        with open(os.path.join(outpath, "scores.txt"), "w") as f:
            f.write(
                f"Run completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write("\n#########################\n\n")

            f.write(f"True Scores: {true_scores}\n")
            f.write(f"Partial Random: {partial_random}\n")
            f.write(f"Full Random: {full_random}\n")
            f.write("\n#########################\n\n")

            f.write(f"Number of latent factors: {self.latent_factors.shape[1]}\n")
            f.write(f"Number of marginals: {len(self.sig_LFs)}\n")
            f.write(f"Number of interactions: {len(self.sig_interacts)}\n")

        with open(os.path.join(outpath, "run_params.txt"), "w") as f:
            f.write(
                f"Run completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

            # Capture show_params output
            output = StringIO()
            with redirect_stdout(output):
                self.show_params()
            f.write(output.getvalue())

    @staticmethod
    def create_summary_table(outpath):
        """
        Create a summary table of the results.
        """
        outs = glob(os.path.join(outpath, "*_out/scores.txt"))
        df = pd.DataFrame(
            columns=[
                "delta",
                "lambda",
                "num_of_LFs",
                "num_of_Sig_LFs",
                "num_of_Interactors",
                "sampleCV_Performance",
            ]
        )

        for out in outs:
            basename = os.path.basename(out.replace("/scores.txt", ""))
            delta_ = basename.split("_")[0]
            lambda_ = basename.split("_")[1]

            with open(out, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "latent factors" in line:
                        num_latent_factors = line.split(": ")[1].strip()
                    if "interactions" in line:
                        num_interactions = line.split(": ")[1].strip()
                    if "marginals" in line:
                        num_marginals = line.split(": ")[1].strip()
                    if "True Scores" in line:
                        true_scores = line.split(": ")[1].strip()

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "delta": [delta_],
                            "lambda": [lambda_],
                            "num_of_LFs": [num_latent_factors],
                            "num_of_Sig_LFs": [num_marginals],
                            "num_of_Interactors": [num_interactions],
                            "sampleCV_Performance": [true_scores],
                        }
                    ),
                ],
                ignore_index=True,
            )

        # df['full_random'] = df['full_random'].astype(float).round(3)
        # df['partial_random'] = df['partial_random'].astype(float).round(3)
        df["sampleCV_Performance"] = pd.to_numeric(
            df["sampleCV_Performance"].astype(str), errors="coerce"
        ).round(3)
        df.sort_values(by="sampleCV_Performance", inplace=True)

        df.to_csv(os.path.join(outpath, "summary_table.csv"), index=False)


class OptimizeSLIDE(SLIDE):
    def __init__(self, input_params, x=None, y=None):
        super().__init__(input_params, x, y)

    def get_latent_factors(
        self,
        x,
        y,
        delta,
        mu=0.5,
        lbd=0.1,
        pure_homo=True,
        verbose=False,
        thresh_fdr=0.2,
        outpath=".",
        love_backend="python",
    ):
        """
        Get the latent factors (aka z_matrix) from the LOVE algorithm.

        Args:
            x (pd.DataFrame): The input data.
            y (pd.Series): The target variable.
            delta (float): The delta parameter.
            mu (float): The mu parameter. Set to 0.5 by default.
            lbd (float): The lambda parameter.
            pure_homo (bool): Whether to use the pure homoscedastic model. Newest LOVE implementation uses False
            verbose (bool): Whether to print verbose output.
            thresh_fdr (float): a numerical constant used for thresholding the correlation matrix to
                                control the false discovery rate
            outpath (str): The path to save the LOVE result.
            love_backend (str): Which LOVE implementation to use: 'python' (default) or 'r'.
        """

        # Standardize X once (LOVE will use this internally, and we need it for Z calculation)
        # Match R's scale(x, T, T) behavior
        x_std = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)

        love_result = call_love(
            X=x_std,
            lbd=lbd,
            mu=mu,
            pure_homo=pure_homo,
            delta=delta,
            verbose=verbose,
            thresh_fdr=thresh_fdr,
            outpath=outpath,
            backend=love_backend,
        )

        # Match R: if LOVE returned None (e.g. Dantzig LP failure), skip this delta
        if love_result is None:
            import logging
            logging.getLogger(__name__).warning(
                f"LOVE returned None for delta={delta} — Dantzig LP solver failed. "
                "Skipping this parameter combination (matches R behavior)."
            )
            self.love_result = None
            return

        self.love_result = love_result
        self.x_std = x_std  # Save for Z matrix calculation

        love_res_path = os.path.join(outpath, "love_result.pkl")
        with open(love_res_path, "wb") as f:
            pickle.dump(self.love_result, f)

        self.A = pd.DataFrame(
            love_result["A"],
            index=x.columns,
            columns=[f"Z{i}" for i in range(love_result["A"].shape[1])],
        )
        self.A.to_csv(os.path.join(outpath, "A.csv"))

        ### Solve for Z matrix from A, Gamma, and C
        self.latent_factors = self.calc_z_matrix(love_result)

        self.latent_factors.to_csv(os.path.join(outpath, "z_matrix.csv"))

    def calc_z_matrix(self, love_result):
        A_hat = love_result["A"]
        Gamma_hat = love_result["Gamma"].copy()  # Make a copy to modify
        C_hat = love_result["C"]
        I_hat = love_result["pureVec"]  # Pure variable indices

        # Use the standardized X that was passed to LOVE
        x = self.x_std.values

        # Compute covariance matrix of standardized X
        Sigma = np.cov(x, rowvar=False)

        # CRITICAL FIX: Estimate Gamma for non-pure variables
        # R SLIDE does this at getLatentFactors.R line 165:
        # Gamma_hat[-I_hat] <- diag(sigma[-I_hat, -I_hat]) - diag(A_hat[-I_hat,] %*% C_hat %*% t(A_hat[-I_hat, ]))
        p = A_hat.shape[0]
        all_indices = set(range(p))
        pure_indices = set(I_hat) if I_hat is not None else set()
        non_pure_indices = list(all_indices - pure_indices)

        if len(non_pure_indices) > 0:
            A_J = A_hat[non_pure_indices, :]
            # Gamma for non-pure = diag(Sigma_JJ) - diag(A_J @ C @ A_J.T)
            Sigma_JJ_diag = np.diag(Sigma[np.ix_(non_pure_indices, non_pure_indices)])
            ACA_diag = np.diag(A_J @ C_hat @ A_J.T)
            Gamma_hat[non_pure_indices] = Sigma_JJ_diag - ACA_diag

        # Replace negative Gamma values (R uses 1e2 for non-pure, 1e-2 for pure)
        Gamma_hat[Gamma_hat < 0] = 1e2

        # Convert Gamma_hat to diagonal matrix and handle zeros
        Gamma_hat = np.where(Gamma_hat == 0, 1e-10, Gamma_hat)
        Gamma_hat_inv = np.diag(Gamma_hat ** (-1))

        # Calculate G_hat matrix
        G_hat = A_hat.T @ Gamma_hat_inv @ A_hat + np.linalg.inv(C_hat)

        # Calculate Z_hat matrix
        Z_hat = x @ Gamma_hat_inv @ A_hat @ np.linalg.pinv(G_hat)

        # Convert to DataFrame with appropriate column names
        Z = pd.DataFrame(
            Z_hat,
            index=self.data.X.index,
            columns=[f"Z{i}" for i in range(Z_hat.shape[1])],
        )
        return Z

    def find_standalone_LFs(
        self,
        latent_factors,
        spec,
        fdr,
        niter,
        f_size,
        n_workers=1,
        knockoff_backend="r",
        knockoff_method="asdp",
        knockoff_shrink=False,
        knockoff_offset=0,
        fstat="glmnet_lambdasmax",
    ):

        machop = Knockoffs(y=self.data.Y.values, z2=latent_factors.values)

        marginal_idxs = machop.select_short_freq(
            z=latent_factors.values,
            y=self.data.Y.values,
            spec=spec,
            fdr=fdr,
            niter=niter,
            f_size=f_size,
            n_workers=n_workers,
            backend=knockoff_backend,
            method=knockoff_method,
            shrink=knockoff_shrink,
            offset=knockoff_offset,
            fstat=fstat,
        )

        self.marginal_idxs = marginal_idxs
        return machop

    def find_interaction_LFs(
        self,
        machop,
        spec,
        fdr,
        niter,
        f_size,
        n_workers=1,
        knockoff_backend="r",
        knockoff_method="asdp",
        knockoff_shrink=False,
        knockoff_offset=0,
        fstat="glmnet_lambdasmax",
        use_r_style=True,
    ):
        """
        Find interacting latent factors using knockoff filter.

        When use_r_style=True (default), uses sequential per-marginal processing that
        matches R's interactionSLIDE.R behavior:
        1. For each marginal, generate interaction terms with ALL other LFs
           (excluding already-used marginals)
        2. Correct y for this marginal's effect (residuals from lm(y ~ z_marginal))
        3. Run knockoff filter on interaction terms vs corrected y
        4. Collect significant interactions

        Parameters
        ----------
        machop : Knockoffs
            Knockoffs object with latent factors.
        spec : float
            Proportion threshold for selection frequency.
        fdr : float
            Target false discovery rate.
        niter : int
            Number of knockoff iterations.
        f_size : int
            Feature size for knockoff subsets.
        n_workers : int
            Number of parallel workers.
        knockoff_backend : str
            Backend for knockoff filter ('r', 'python', 'knockpy').
        knockoff_method : str
            Knockoff construction method.
        knockoff_shrink : bool
            Whether to use covariance shrinkage.
        knockoff_offset : int
            Knockoff procedure offset (0 or 1).
        fstat : str
            Feature statistic method.
        use_r_style : bool
            If True (default), use R-style sequential per-marginal processing.
            If False, use legacy batch processing.
        """
        if not use_r_style:
            # Legacy batch processing
            self._find_interaction_LFs_batch(
                machop,
                spec,
                fdr,
                niter,
                f_size,
                n_workers,
                knockoff_backend,
                knockoff_method,
                knockoff_shrink,
                knockoff_offset,
                fstat,
            )
            return

        # R-style sequential per-marginal processing
        all_sig_interactions = []  # List of (marginal_idx, interacting_idx) tuples
        used_marginals = set()

        z_all = self.latent_factors.values  # Full LF matrix
        y = self.data.Y.values
        n_lfs = z_all.shape[1]

        for marg_idx in self.marginal_idxs:
            used_marginals.add(marg_idx)
            z_marginal = z_all[:, marg_idx]

            # Get candidate columns: all LFs except already-used marginals
            candidate_idxs = [i for i in range(n_lfs) if i not in used_marginals]
            if len(candidate_idxs) == 0:
                continue

            z_candidates = z_all[:, candidate_idxs]

            # Create interaction terms: marginal × each candidate
            interaction_terms = (
                z_marginal[:, np.newaxis] * z_candidates
            )  # (n, n_candidates)

            # Correct y for marginal effect (residuals from lm(y ~ z_marginal))
            corrected_y = Knockoffs.correct_y(z_marginal, y)

            # Run knockoff filter
            sig_interaction_local_idxs = Knockoffs.select_short_freq(
                z=interaction_terms,
                y=corrected_y,
                spec=spec,
                fdr=fdr,
                niter=niter,
                f_size=f_size,
                n_workers=n_workers,
                backend=knockoff_backend,
                method=knockoff_method,
                shrink=knockoff_shrink,
                offset=knockoff_offset,
                fstat=fstat,
            )

            # Map back to original LF indices
            for local_idx in sig_interaction_local_idxs:
                interacting_idx = candidate_idxs[local_idx]
                all_sig_interactions.append((marg_idx, interacting_idx))

        # Store results
        if len(all_sig_interactions) == 0:
            self.interaction_pairs = np.array([])
            self.interaction_terms = np.array([])
        else:
            pairs = np.array(all_sig_interactions).T  # Shape: (2, n_pairs)
            self.interaction_pairs = pairs
            # Reconstruct interaction term values
            self.interaction_terms = np.column_stack(
                [z_all[:, m] * z_all[:, j] for m, j in all_sig_interactions]
            )

    def _find_interaction_LFs_batch(
        self,
        machop,
        spec,
        fdr,
        niter,
        f_size,
        n_workers=1,
        knockoff_backend="r",
        knockoff_method="asdp",
        knockoff_shrink=False,
        knockoff_offset=0,
        fstat="glmnet_lambdasmax",
    ):
        """
        Legacy batch interaction detection (original Python implementation).

        This tests all marginal×non-marginal interactions simultaneously without
        y-correction, which may produce fewer interactions than R's approach.
        """
        machop.add_z1(marginal_idxs=self.marginal_idxs)

        if machop.z2.shape[1] == 0:
            print(
                "All LFs are standalone, consider lowering delta for LOVE to find more LFs"
            )
            self.interaction_pairs = np.array([])
            self.interaction_terms = np.array([])
            return

        # Flatten interaction terms for knockoff selection
        interaction_terms = machop.interaction_terms.reshape(machop.n, -1)

        # Get significant interactions from flattened array
        sig_interactions = machop.select_short_freq(
            z=interaction_terms,
            y=self.data.Y.values,
            spec=spec,
            fdr=fdr,
            niter=niter,
            f_size=f_size,
            n_workers=n_workers,
            backend=knockoff_backend,
            method=knockoff_method,
            shrink=knockoff_shrink,
            offset=knockoff_offset,
            fstat=fstat,
        )

        if len(sig_interactions) == 0:
            self.interaction_pairs = np.array([])
            self.interaction_terms = np.array([])
        else:
            n_candidates = machop.z2.shape[1]
            marginal_lf = self.marginal_idxs[sig_interactions // n_candidates]
            z2_cols = np.array(
                [
                    i
                    for i in range(self.latent_factors.shape[1])
                    if i not in self.marginal_idxs
                ]
            )

            assert len(z2_cols) == n_candidates, (
                "Number of candidates does not match (implementation error)"
            )

            interacting_lf = z2_cols[sig_interactions % n_candidates]
            self.interaction_pairs = np.array([marginal_lf, interacting_lf])
            self.interaction_terms = interaction_terms[:, sig_interactions]

    def run_SLIDE(
        self,
        latent_factors,
        niter,
        spec,
        fdr,
        verbose=False,
        n_workers=1,
        outpath=".",
        do_interacts=True,
        knockoff_backend="python",
        knockoff_method="asdp",
        knockoff_shrink=False,
        knockoff_offset=0,
        fstat="glmnet_lambdasmax",
    ):

        f_size = self.calc_default_fsize(latent_factors.shape[1])

        if verbose:
            print(f"Calculated f_size: {f_size}")
            print(f"Knockoff backend: {knockoff_backend}")
            if knockoff_backend in ("python", "knockpy"):
                print(f"Knockoff method: {knockoff_method}")
                print(f"Knockoff shrink: {knockoff_shrink}")
                print(f"Knockoff offset: {knockoff_offset}")
            print(f"Feature statistic: {fstat}")
            print(f"Finding standalone LF...")

        ### Find standalone LFs
        machop = self.find_standalone_LFs(
            latent_factors,
            spec,
            fdr,
            niter,
            f_size,
            n_workers,
            knockoff_backend=knockoff_backend,
            knockoff_method=knockoff_method,
            knockoff_shrink=knockoff_shrink,
            knockoff_offset=knockoff_offset,
            fstat=fstat,
        )

        if len(self.marginal_idxs) == 0:
            print("No standalone LF found")
            self.sig_LFs = []
            self.sig_interacts = []
            return

        self.sig_LFs = [f"Z{i}" for i in self.marginal_idxs]
        np.savetxt(os.path.join(outpath, "sig_LFs.txt"), self.sig_LFs, fmt="%s")

        if verbose:
            print(f"Found {len(self.marginal_idxs)} standalone LF")

        ### Find interacting LFs

        if do_interacts:
            if verbose:
                print(f"Finding interacting LF...")

            self.find_interaction_LFs(
                machop,
                spec,
                fdr,
                niter,
                f_size,
                n_workers,
                knockoff_backend=knockoff_backend,
                knockoff_method=knockoff_method,
                knockoff_shrink=knockoff_shrink,
                knockoff_offset=knockoff_offset,
                fstat=fstat,
            )

            if verbose:
                print(f"Found {len(self.interaction_pairs)} interacting LF")

            self.sig_interacts = [f"Z{j}" for i, j in self.interaction_pairs.T]
            np.savetxt(
                os.path.join(outpath, "sig_interacts.txt"), self.sig_interacts, fmt="%s"
            )

        else:
            self.sig_interacts = []

    def run_pipeline(self, verbose=True, n_workers=1, rerun=False):

        if verbose:
            self.show_params()

        for delta_iter, lambda_iter in product(
            self.input_params["delta"], self.input_params["lambda"]
        ):
            out_iter = os.path.join(
                self.input_params["out_path"], f"{delta_iter}_{lambda_iter}_out"
            )

            if rerun and os.path.exists(out_iter):
                self.load_state(out_iter)

                try:
                    self.data.Y = self.adata.Y.loc[self.latent_factors.index]
                except:
                    print("input data obs do not match previous results")
                    continue

                try:
                    self.data.X = self.data.X[self.A.columns]
                except:
                    print("input data vars do not match previous results")
                    continue

            else:
                os.makedirs(out_iter, exist_ok=True)

                if verbose:
                    print(
                        f"Running LOVE with delta={delta_iter} and lambda={lambda_iter}"
                    )

                try:
                    self.get_latent_factors(
                        x=self.data.X,
                        y=self.data.Y,
                        delta=delta_iter,
                        lbd=lambda_iter,
                        thresh_fdr=self.input_params["thresh_fdr"],
                        pure_homo=self.input_params["pure_homo"],
                        verbose=verbose,
                        outpath=out_iter,
                        love_backend=self.input_params.get("love_backend", "python"),
                    )

                    # Handle LOVE returning None (Dantzig LP failure, matches R)
                    if self.love_result is None:
                        if verbose:
                            print(f"LOVE returned None for delta={delta_iter}, "
                                  f"lambda={lambda_iter} — skipping.")
                        continue

                    if verbose:
                        print(
                            f"LOVE found {self.latent_factors.shape[1]} latent factors."
                        )

                except Exception as e:
                    print(f"\nError running LOVE: {e}\n")
                    print("##################\n")

                    continue

                if verbose:
                    print("\nRunning SLIDE knockoffs...")

                self.run_SLIDE(
                    latent_factors=self.latent_factors,
                    niter=self.input_params["niter"],
                    spec=self.input_params["spec"],
                    fdr=self.input_params["fdr"],
                    n_workers=self.input_params["n_workers"],
                    verbose=verbose,
                    outpath=out_iter,
                    do_interacts=self.input_params["do_interacts"],
                    knockoff_backend=self.input_params.get("knockoff_backend", "r"),
                    knockoff_method=self.input_params.get("knockoff_method", "asdp"),
                    knockoff_shrink=self.input_params.get("knockoff_shrink", False),
                    knockoff_offset=self.input_params.get("knockoff_offset", 0),
                    fstat=self.input_params.get("fstat", "glmnet_lambdasmax"),
                )

                if verbose:
                    print("\nSLIDE complete.")

            if len(self.marginal_idxs) > 0:
                sig_LF_genes = {
                    str(lf): SLIDE.get_LF_genes(
                        A=self.A,
                        X=self.data.X,
                        lf=lf,
                        y=self.data.Y,
                        top_feats=self.input_params["SLIDE_top_feats"],
                        outpath=out_iter,
                    )
                    for lf in self.sig_LFs
                }
                Plotter.plot_latent_factors(
                    sig_LF_genes, outdir=out_iter, title="marginal_LFs"
                )

                sig_interact_genes = {
                    str(lf): SLIDE.get_LF_genes(
                        A=self.A,
                        X=self.data.X,
                        lf=lf,
                        y=self.data.Y,
                        top_feats=self.input_params["SLIDE_top_feats"],
                        outpath=out_iter,
                    )
                    for lf in self.sig_interacts
                }

                if len(sig_interact_genes) > 0:
                    Plotter.plot_latent_factors(
                        sig_interact_genes, outdir=out_iter, title="interaction_LFs"
                    )
                    Plotter.plot_interactions(
                        self.interaction_pairs,
                        outdir=out_iter,
                        title="interaction_pairs",
                    )

                scores = SLIDE_Estimator.score_performance(
                    latent_factors=self.latent_factors,
                    sig_LFs=self.sig_LFs,
                    sig_interacts=self.sig_interacts,
                    y=self.data.Y,
                    n_iters=2000,
                    test_size=0.15,
                    scaler="standard",
                )

                Plotter.plot_controlplot(scores, outdir=out_iter, title="control_plot")

                Plotter.plot_corr_network(
                    self.data.X,
                    lf_dict=sig_LF_genes | sig_interact_genes,
                    outdir=out_iter,
                )

            else:
                scores = None
                self.sig_interacts = []

            self.save_params(out_iter, scores)

            if verbose:
                print(f"\nCompleted {delta_iter}_{lambda_iter}\n")
                print("##################\n")

        self.create_summary_table(self.input_params["out_path"])
