using Test

include("utils.jl")

@testset "utils.jl" begin
    
    @testset "iterateOverVectors(f, x)" begin
        
        x = [0 0 2; 1 1 1; 0 2 2]

        res = iterateOverVectors(sum, x)

        @test res[1] == 1
        @test res[2] == 3
        @test res[3] == 5

    end

end