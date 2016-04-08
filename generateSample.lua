--this was obtained via personal communication with alejandro newell

function generateSample(set, idx)
    local pts = annot[set]['part']
    local c = annot[set]['center']
    local s = annot[set]['scale']
    local imgName = annot[set]['images'][idx]
    local idxs = annot[set]['imageToIdxs'][imgName]

    local flip = false
    if torch.uniform() < .5 then flip = true end
    local img = image.load(opt.dataDir .. '/images/' .. imgName)

    -- For pose estimation with a centered/scaled figure
    local inp = crop(img, c[idx], s[idx], 0, opt.inputRes)
    local out = torch.zeros(unpack(labelDim))
    for _,i in ipairs(idxs) do
        for j = 1,nParts do
            local pt = pts[i][j]
            if pt[1] > 0 then
                drawGaussian(out[j], transform(pt, c[idx], s[idx], 0, opt.outputRes), 2)
            end
        end
    end

    if flip then
        inp = image.hflip(inp)
        out = image.hflip(out)
        shuffleLR(out)
    end

    return inp,out
end
