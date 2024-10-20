"""
    iterateOverVectors(f, x)

Apply the given function to the given matrix, vector by vector.

# Arguments
- `f::Function`: function which takes p-vectors.
- `x::Matrix`: p x q matrix.
"""
function iterateOverVectors(f::Function, x::Matrix)
    return mapslices(f, x, dims=1)
end