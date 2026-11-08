# Run from the repository root with the configured R environment.
description <- read.dcf('DESCRIPTION')
fields <- intersect(c('Depends', 'Imports', 'Suggests'), colnames(description))
declared <- trimws(gsub('\\s*\\([^)]*\\)', '',
  unlist(strsplit(paste(description[1, fields], collapse = ','), ','))))
stopifnot(!'EssReg' %in% declared)
for (path in list.files('R', pattern = '\\.R$', full.names = TRUE)) {
  code <- paste(deparse(parse(path, keep.source = FALSE)), collapse = '\n')
  namespaces <- unique(sub('::+$', '',
    regmatches(code, gregexpr('[A-Za-z][A-Za-z0-9.]*::+', code))[[1]]))
  missing <- setdiff(namespaces, c(declared, 'base', 'parallel', 'tools', 'SLIDE'))
  if (length(missing)) stop(path, ': undeclared dependencies: ', paste(missing, collapse = ', '))
}

e <- new.env(parent = globalenv())
e$`%>%` <- magrittr::`%>%`
for (name in c('standCV', 'predZ', 'readER', 'recoverGroup', 'pureRowInd')) {
  sys.source(file.path('R', paste0(name, '.R')), e)
}
train <- matrix(c(1, 2, 4, 8, 2, 5, 6, 9), ncol = 2)
valid <- matrix(c(10, 20, 11, 22), ncol = 2)
y <- c(1, 3, 2, 7)
standardized <- e$standCV(train_x = train, valid_x = valid, train_y = y, valid_y = c(8, 10))
expected <- sweep(sweep(valid, 2, colMeans(train), '-'), 2, apply(train, 2, sd), '/')
stopifnot(isTRUE(all.equal(standardized$train_x, scale(train))),
  isTRUE(all.equal(standardized$valid_x, expected)),
  isTRUE(all.equal(as.numeric(standardized$valid_y), (c(8, 10) - mean(y)) / sd(y))))
stopifnot(isTRUE(all.equal(e$predZ(x = train,
  er_res = list(A = diag(2), C = diag(2), Gamma = c(1, 1))), train / 2)))
x <- matrix(1:12, ncol = 3, dimnames = list(NULL, c('a', 'b', 'c')))
groups <- e$readER(x = x, er_res = list(A = rbind(c(1, 0), c(0, -1), c(.5, .5))))
stopifnot(identical(names(groups$pure_vars), c('a', 'b')),
  identical(names(groups$mix_vars), 'c'),
  identical(names(groups$clusters$Z1$pos), c('a', 'c')),
  identical(names(groups$clusters$Z2$neg), 'b'))
message('PASS: declared namespace dependencies and local standardization, projection and loading-group contracts')
