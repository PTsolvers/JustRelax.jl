_f(x::Int) = x^1.2
_f(x::Float64) = x^2.1

function f1(x...)
    sum(_f(xi) for xi in x)
end

function f2(x::Vararg{Any, N}) where N
    sum(_f(xi) for xi in x)
end

x1 = 1, 123.3, 21, 3.2 
x2 = 1.8, 3, 2.1, 3 

@code_typed f1(x1...)
@code_typed f1(x2...)

@code_typed f2(x2...)
@code_typed f2(x1...)

@btime f1($x1...)
@btime f2($x1...)

@btime f1($x2...)
@btime f2($x2...)