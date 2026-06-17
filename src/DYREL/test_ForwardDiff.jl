using ForwardDiff

function testfunc_scalar_ouput(x)
    return sin(x[1]) + x[2]^2
end

function testfunc_vector_output(x,y)
    return [sin(x[1]) + x[2]^2 + y[1]]
end

x = [1.0,2.0]
y = [5.0]
res = testfunc_scalar_ouput(x)

ForwardDiff.gradient(testfunc_scalar_ouput, x)

ForwardDiff.jacobian(x-> testfunc_vector_output(x,y), x)
ForwardDiff.jacobian(y-> testfunc_vector_output(x,y), y)