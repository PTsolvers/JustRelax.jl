@views function no_slip!(Ax, Ay, bc)
    if bc.left
        Ax[1, :] .= 0
        Ay[1, :] .= -Ay[2, :]
    end
    if bc.right
        Ax[end, :] .= 0
        Ay[end, :] .= -Ay[end - 1, :]
    end
    if bc.bot
        Ax[:, 2] .= Ax[:, 3] / 3
        Ax[:, 1] .= -Ax[:, 2]
        Ay[:, 1] .= 0
    end
    if bc.top
        Ax[:, end] .= -Ax[:, end - 1]
        Ay[:, end] .= 0
    end
end

@views function no_slip!(Ax, Ay, Az, bc)
    if bc.left
        Ax[1, :, :] .= 0
        Ay[1, :, :] .= -Ay[2, :, :]
        Az[1, :, :] .= -Az[2, :, :]
    end
    if bc.right
        Ax[end, :, :] .= 0
        Ay[end, :, :] .= -Ay[end - 1, :, :]
        Az[end, :, :] .= -Az[end - 1, :, :]
    end

    if bc.front
        Ax[:, 1, :] .= -Ax[:, 2, :]
        Ay[:, 1, :] .= 0
        Az[:, 1, :] .= -Az[:, 2, :]
    end

    if bc.back
        Ax[:, end, :] .= -Ax[:, end - 1, :]
        Ay[:, end, :] .= 0
        Az[:, end, :] .= -Az[:, end - 1, :]
    end

    if bc.bot
        Ax[:, :, 1] .= -Ax[:, :, 2]
        Ay[:, :, 1] .= -Ay[:, :, 2]
        Az[:, :, 1] .= 0
    end
    if bc.top
        Ax[:, :, end] .= -Ax[:, :, end - 1]
        Ay[:, :, end] .= -Ay[:, :, end - 1]
        Az[:, :, end] .= 0
    end
end
