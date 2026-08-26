@parallel_indices (i) function _no_slip_left!(Ax, Ay)
    @inbounds begin
        i ≤ size(Ax, 2) && (Ax[1, i] = 0)
        i ≤ size(Ay, 2) && (Ay[1, i] = -Ay[2, i])
    end
    return nothing
end

@parallel_indices (i) function _no_slip_right!(Ax, Ay)
    @inbounds begin
        i ≤ size(Ax, 2) && (Ax[end, i] = 0)
        i ≤ size(Ay, 2) && (Ay[end, i] = -Ay[end - 1, i])
    end
    return nothing
end

@parallel_indices (i) function _no_slip_bot!(Ax, Ay)
    @inbounds begin
        i ≤ size(Ax, 1) && (Ax[i, 1] = -Ax[i, 2])
        i ≤ size(Ay, 1) && (Ay[i, 1] = 0)
    end
    return nothing
end

@parallel_indices (i) function _no_slip_top!(Ax, Ay)
    @inbounds begin
        i ≤ size(Ax, 1) && (Ax[i, end] = -Ax[i, end - 1])
        i ≤ size(Ay, 1) && (Ay[i, end] = 0)
    end
    return nothing
end

function no_slip!(Ax, Ay, bc)
    bc.left && (@parallel (@idx max(size(Ax, 2), size(Ay, 2))) _no_slip_left!(Ax, Ay))
    bc.right && (@parallel (@idx max(size(Ax, 2), size(Ay, 2))) _no_slip_right!(Ax, Ay))
    bc.bot && (@parallel (@idx max(size(Ax, 1), size(Ay, 1))) _no_slip_bot!(Ax, Ay))
    bc.top && (@parallel (@idx max(size(Ax, 1), size(Ay, 1))) _no_slip_top!(Ax, Ay))
    return nothing
end

@parallel_indices (i, j) function _no_slip_left!(Ax, Ay, Az)
    @inbounds begin
        i ≤ size(Ax, 2) && j ≤ size(Ax, 3) && (Ax[1, i, j] = 0)
        i ≤ size(Ay, 2) && j ≤ size(Ay, 3) && (Ay[1, i, j] = -Ay[2, i, j])
        i ≤ size(Az, 2) && j ≤ size(Az, 3) && (Az[1, i, j] = -Az[2, i, j])
    end
    return nothing
end

@parallel_indices (i, j) function _no_slip_right!(Ax, Ay, Az)
    @inbounds begin
        i ≤ size(Ax, 2) && j ≤ size(Ax, 3) && (Ax[end, i, j] = 0)
        i ≤ size(Ay, 2) && j ≤ size(Ay, 3) && (Ay[end, i, j] = -Ay[end - 1, i, j])
        i ≤ size(Az, 2) && j ≤ size(Az, 3) && (Az[end, i, j] = -Az[end - 1, i, j])
    end
    return nothing
end

@parallel_indices (i, j) function _no_slip_front!(Ax, Ay, Az)
    @inbounds begin
        i ≤ size(Ax, 1) && j ≤ size(Ax, 3) && (Ax[i, 1, j] = -Ax[i, 2, j])
        i ≤ size(Ay, 1) && j ≤ size(Ay, 3) && (Ay[i, 1, j] = 0)
        i ≤ size(Az, 1) && j ≤ size(Az, 3) && (Az[i, 1, j] = -Az[i, 2, j])
    end
    return nothing
end

@parallel_indices (i, j) function _no_slip_back!(Ax, Ay, Az)
    @inbounds begin
        i ≤ size(Ax, 1) && j ≤ size(Ax, 3) && (Ax[i, end, j] = -Ax[i, end - 1, j])
        i ≤ size(Ay, 1) && j ≤ size(Ay, 3) && (Ay[i, end, j] = 0)
        i ≤ size(Az, 1) && j ≤ size(Az, 3) && (Az[i, end, j] = -Az[i, end - 1, j])
    end
    return nothing
end

@parallel_indices (i, j) function _no_slip_bot!(Ax, Ay, Az)
    @inbounds begin
        i ≤ size(Ax, 1) && j ≤ size(Ax, 2) && (Ax[i, j, 1] = -Ax[i, j, 2])
        i ≤ size(Ay, 1) && j ≤ size(Ay, 2) && (Ay[i, j, 1] = -Ay[i, j, 2])
        i ≤ size(Az, 1) && j ≤ size(Az, 2) && (Az[i, j, 1] = 0)
    end
    return nothing
end

@parallel_indices (i, j) function _no_slip_top!(Ax, Ay, Az)
    @inbounds begin
        i ≤ size(Ax, 1) && j ≤ size(Ax, 2) && (Ax[i, j, end] = -Ax[i, j, end - 1])
        i ≤ size(Ay, 1) && j ≤ size(Ay, 2) && (Ay[i, j, end] = -Ay[i, j, end - 1])
        i ≤ size(Az, 1) && j ≤ size(Az, 2) && (Az[i, j, end] = 0)
    end
    return nothing
end

function no_slip!(Ax, Ay, Az, bc)
    bc.left && (@parallel (@idx (max(size(Ax, 2), size(Ay, 2), size(Az, 2)), max(size(Ax, 3), size(Ay, 3), size(Az, 3)))) _no_slip_left!(Ax, Ay, Az))
    bc.right && (@parallel (@idx (max(size(Ax, 2), size(Ay, 2), size(Az, 2)), max(size(Ax, 3), size(Ay, 3), size(Az, 3)))) _no_slip_right!(Ax, Ay, Az))
    bc.front && (@parallel (@idx (max(size(Ax, 1), size(Ay, 1), size(Az, 1)), max(size(Ax, 3), size(Ay, 3), size(Az, 3)))) _no_slip_front!(Ax, Ay, Az))
    bc.back && (@parallel (@idx (max(size(Ax, 1), size(Ay, 1), size(Az, 1)), max(size(Ax, 3), size(Ay, 3), size(Az, 3)))) _no_slip_back!(Ax, Ay, Az))
    bc.bot && (@parallel (@idx (max(size(Ax, 1), size(Ay, 1), size(Az, 1)), max(size(Ax, 2), size(Ay, 2), size(Az, 2)))) _no_slip_bot!(Ax, Ay, Az))
    bc.top && (@parallel (@idx (max(size(Ax, 1), size(Ay, 1), size(Az, 1)), max(size(Ax, 2), size(Ay, 2), size(Az, 2)))) _no_slip_top!(Ax, Ay, Az))
    return nothing
end
