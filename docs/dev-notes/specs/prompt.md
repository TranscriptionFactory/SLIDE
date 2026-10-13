  ## Context
  **Note:** serena is installed in this venv `.venv_utils/bin/activate`                                                                                                 
  - Specs live in specs/: design.md (architecture), specs/specs/*/spec.md (per-component contracts),        
  initial_spec_guidance.md (index)                                                                          
  - The Python reference implementation is in this repo — use it to understand behavior, not as a style     
  guide                                                                                                
  - The Rust port targets numerical correctness against R fixtures (see specs/specs/correctness/spec.md for 
  tolerances)                                                                                              
                                                                                                            
  ## Task tracking                                                                                        
  - Read specs/tasks.md at session start for current status                                                 
  - Before starting work, update the task you're working on to "in progress"                              
  - After completing work, mark the task "done" with a brief note on what was delivered                     
  - If you discover new work, add it to tasks.md in the appropriate section                                 
                                                                                                            
  ## Workflow                                                                                               
  1. Run `git log --oneline -10 && git status` to recover context                                           
  2. Read specs/tasks.md to find the next unblocked task                                                    
  3. Read the relevant spec(s) before writing any code                                                      
  4. Implement incrementally — one function or module at a time
  5. Write tests alongside implementation (unit tests inline, integration tests in tests/)                  
  6. Commit after each meaningful unit of work with message format:                                         
     - Line 1: What changed                                                                                 
     - Line 2: Why / intent                                                                                 
     - Line 3: Next step or blocker (if relevant)                                                           
  7. Update specs/tasks.md after each commit                                                                
  8. If blocked or uncertain, stop and explain rather than guessing                                         
                                                                                                            
  ## Standards                                                                                              
  - Follow specs/design.md for crate structure, library choices, error handling                           
  - Match R numerical behavior per correctness/spec.md tolerances                                           
  - No unwrap/expect in library code; Result everywhere                                                     
  - Use rayon for coarse-grained parallelism (fold-level, seed-level), not within matrix ops                
  - Prefer nalgebra for decompositions, ndarray for slicing/broadcasting   