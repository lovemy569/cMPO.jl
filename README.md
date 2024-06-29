# cMPO

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lovemy569.github.io/cMPO.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lovemy569.github.io/cMPO.jl/dev/)
[![Build Status](https://github.com/lovemy569/cMPO.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lovemy569/cMPO.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lovemy569/cMPO.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lovemy569/cMPO.jl)

I try to reproduce the results of the paper [Tang W, Tu H H, Wang L. Continuous matrix product operator approach to finite temperature quantum states[J]. Physical Review Letters, 2020, 125(17): 170604](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.170604) in Julia.

You can change and run the datagenerate related parameters in src to obtain the cMPS of the XXZ Heisenberg chain, and based on this calculate the relevant observables in the thermodynamic limit.

I have uploaded some raw data in the data folder, which can be processed by dataprocess to obtain FIG.2 and FIG.S3 in the article.

It is not clear whether similar results can be obtained with higher key dimensions (e.g. 30). If there are any errors or discoveries in the future, this code base will be updated. As the first completed Julia code, thanks to [GiggleLiu](https://github.com/GiggleLiu) for his guidance, the [Python source code](https://github.com/TensorBFS/cMPO) provided by the author of the article, and [Sharon-Liang/JuliaCMPO](https://github.com/Sharon-Liang/JuliaCMPO) for reproducing this article before.

For more details, please refer to document.


