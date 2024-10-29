using Gadfly
"""
    plotApproxVSMCMC(approxDensity, MCMCsample; a, b, xLabel, yLabel)

Plot the approximative density function against the sampled posterior density obtained by MCMC.

# Arguments
- `approxDensity::Function`: approximative density function (univariate).
- `MCMCsample::DenseVector`: samples of the posterior density obtained by MCMC.
- `a::Real`: beginning of the x axis.
- `b::Real`: end of the x axis.
- `step::Real`: resolution of the plot.
- `xLabel::String`: label of the x axis.
- `yLabel::String`: label of the y axis.
"""
function plotApproxVSMCMC(
    approxDensity::Function,
    MCMCsample::DenseVector;
    a::Real,
    b::Real,
    step::Real,
    xLabel::String,
    yLabel::String,
)
    
    x = a:step:b;

    plot(
        layer(x=x, y=approxDensity.(x), Geom.line, Theme(default_color="red")),
        layer(x=MCMCsample, Geom.histogram(density=true)),
        Guide.manual_color_key("Legend", ["Approximation", "Posteriori"], ["red", "deepskyblue"]),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC"),
        Guide.xlabel(xLabel),
        Guide.ylabel(yLabel),
    )
end


"""
    plotApproxVSMCMC(approxDensity, MCMCsample, path; a, b, xLabel, yLabel)

Plot AND STORE the approximative density function against the sampled posterior density obtained by MCMC.

# Arguments
- `approxDensity::Function`: approximative density function (univariate).
- `MCMCsample::DenseVector`: samples of the posterior density obtained by MCMC.
- `path::String`: where to store the plot.
- `a::Real`: beginning of the x axis.
- `b::Real`: end of the x axis.
- `step::Real`: resolution of the plot.
- `xLabel::String`: label of the x axis.
- `yLabel::String`: label of the y axis.
"""
function plotApproxVSMCMC(
    approxDensity::Function,
    MCMCsample::DenseVector,
    path::String;
    a::Real,
    b::Real,
    step::Real,
    xLabel::String,
    yLabel::String,
)
    
    x = a:step:b;

    p = plot(
        layer(x=x, y=approxDensity.(x), Geom.line, Theme(default_color="red")),
        layer(x=MCMCsample, Geom.histogram(density=true)),
        Guide.manual_color_key("Legend", ["Approximation", "Posteriori"], ["red", "deepskyblue"]),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC"),
        Guide.xlabel(xLabel),
        Guide.ylabel(yLabel),
    )

    draw(PNG(path, dpi=300), p)
end