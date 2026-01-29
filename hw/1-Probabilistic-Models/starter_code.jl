import DMUStudent.HW1

#------------- 
# Problem 4
#-------------

function f(a, bs)

    res = Vector{Vector{eltype(a)}}()
    max = Vector{eltype(a)}()
    
    for b in bs
        push!(res,a * b)
    end

    if length(res) == 1
        return res[1]
    end

    res_matrix =  hcat(res...)

    for row in eachrow(res_matrix)
        push!(max,maximum(row))
    end

    return max
end

# You can can test it yourself with inputs like this
a = [1.0 2.0; 3.0 4.0]
@show a
bs = [[1.0, 2.0]]
@show bs
@show f(a, bs)

# This is how you create the json file to submit
HW1.evaluate(f, "bijan.jourabchi@colorado.edu")
