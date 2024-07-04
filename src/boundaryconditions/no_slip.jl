@parallel_indices (i) function no_slip1!(Ax, Ay, bc)
    @inbounds begin
        if bc.left
            (1 < i < size(Ay, 2)) && (Ay[1, i] = -Ay[2, i])
        end
        if bc.right
            (1 < i < size(Ay, 2)) && (Ay[end, i] = -Ay[end - 1, i])
        end
        if bc.bot
            (1 < i < size(Ax, 1)) && (Ax[i, 1] = -Ax[i, 2])
        end
        if bc.top
            (1 < i < size(Ax, 1)) && (Ax[i, end] = -Ax[i, end - 1])
        end
    end
    return nothing
end

@parallel_indices (i) function no_slip2!(Ax, Ay, bc)
    @inbounds begin
        if bc.left
            (i ≤ size(Ax, 2)) && (Ax[1, i] = 0.0)
        end
        if bc.right
            (i ≤ size(Ax, 2)) && (Ax[end, i] = 0.0)
        end
        if bc.bot
            (i ≤ size(Ay, 1)) && (Ay[i, 1] = 0.0)
        end
        if bc.top
            (i ≤ size(Ay, 1)) && (Ay[i, end] = 0.0)
        end
    end
    return nothing
end

@parallel_indices (i, j) function no_slip1!(Ax, Ay, Az, bc)
    @inbounds begin
        if bc.left
            (1 ≤ size(Ay, 2)) && (1 ≤ size(Ay, 3)) && (Ay[1, i, j] = -Ay[2, i, j])
            (1 ≤ size(Az, 2)) && (1 ≤ size(Az, 3)) && (Az[1, i, j] = -Az[2, i, j])
        end
        if bc.right
            (1 ≤ size(Ay, 2)) && (1 ≤ size(Ay, 3)) && (Ay[end, i, j] = -Ay[end - 1, i, j])
            (1 ≤ size(Az, 2)) && (1 ≤ size(Az, 3)) && (Az[end, i, j] = -Az[end - 1, i, j])
        end

        if bc.front
            (1 ≤ size(Ax, 1)) && (1 ≤ size(Ax, 3)) && (Ax[i, 1, j] = -Ax[i, 2, j])
            (1 ≤ size(Az, 1)) && (1 ≤ size(Az, 3)) && (Az[i, 1, j] = -Az[i, 2, j])
        end
        if bc.back
            (1 ≤ size(Ax, 1)) && (1 ≤ size(Ax, 3)) && (Ax[i, end, j] = -Ax[i, end - 1, j])
            (1 ≤ size(Az, 1)) && (1 ≤ size(Az, 3)) && (Az[i, end, j] = -Az[i, end - 1, j])
        end

        if bc.bot
            (1 ≤ size(Ax, 1)) && (1 ≤ size(Ax, 2)) && (Ax[i, j, 1] = -Ax[i, j, 2])
            (1 ≤ size(Ay, 1)) && (1 ≤ size(Ay, 2)) && (Ay[i, j, 1] = -Ay[i, j, 2])
        end
        if bc.top
            (1 ≤ size(Ax, 1)) && (1 ≤ size(Ax, 2)) && (Ax[i, j, end] = -Ax[i, j, end - 1])
            (1 ≤ size(Ay, 1)) && (1 ≤ size(Ay, 2)) && (Ay[i, j, end] = -Ay[i, j, end - 1])
        end
    end
    return nothing
end

@parallel_indices (i, j) function no_slip2!(Ax, Ay, Az, bc)
    @inbounds begin
        ##
        if bc.left
            (i ≤ size(Ax, 2)) && (j ≤ size(Ax, 3)) && (Ax[1, i, j] = 0.0)
        end
        if bc.right
            (i ≤ size(Ax, 2)) && (j ≤ size(Ax, 3)) && (Ax[end, i, j] = 0.0)
        end

        if bc.front
            (i ≤ size(Ay, 1)) && (i ≤ size(Ay, 3)) && (Ay[i, 1, j] = 0.0)
        end
        if bc.back
            (i ≤ size(Ay, 1)) && (i ≤ size(Ay, 3)) && (Ay[i, end, j] = 0.0)
        end

        if bc.bot
            (i ≤ size(Az, 1)) && (j ≤ size(Az, 2)) && (Az[i, j, 1] = 0.0)
        end
        if bc.top
            (i ≤ size(Az, 1)) && (j ≤ size(Az, 2)) && (Az[i, j, end] = 0.0)
        end
    end
    return nothing
end
