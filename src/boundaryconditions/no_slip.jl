@parallel_indices (i) function no_slip!(Ax, Ay, bc)
    @inbounds begin
        if bc.left
            (i ≤ size(Ax, 2)) && (Ax[1, i] = 0.0)
            (1 < i < size(Ay, 2)) && (Ay[1, i] = -Ay[2, i])
        end
        if bc.right
            (i ≤ size(Ax, 2)) && (Ax[end, i] = 0.0)
            (1 < i < size(Ay, 2)) && (Ay[end, i] = -Ay[end - 1, i])
        end
        if bc.bot
            (i ≤ size(Ay, 1)) && (Ay[i, 1] = 0.0)
            (1 < i < size(Ax, 1)) && (Ax[i, 1] = -Ax[i, 2])
        end
        if bc.top
            (i ≤ size(Ay, 1)) && (Ay[i, end] = 0.0)
            (1 < i < size(Ax, 1)) && (Ax[i, end] = -Ax[i, end - 1])
        end
        # corners
        # bc.bot && (Ax[1, 1] = 0.0; Ax[1, 1] = 0.0)
        # bc.left && bc.bot && (Ax[1, 1] = 0.0)
        # bc.right && bc.top && (Ay[end, end] = 0.0)
    end
    return nothing
end