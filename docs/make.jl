using cMPO
using Documenter

DocMeta.setdocmeta!(cMPO, :DocTestSetup, :(using cMPO); recursive=true)

makedocs(;
    modules=[cMPO],
    authors="haojie",
    sitename="cMPO.jl",
    format=Documenter.HTML(;
        canonical="https://lovemy569.github.io/cMPO.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lovemy569/cMPO.jl",
    devbranch="main",
)
