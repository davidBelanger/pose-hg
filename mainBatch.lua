require 'paths'
paths.dofile('util.lua')
paths.dofile('img.lua')

--------------------------------------------------------------------------------
-- Initialization
--------------------------------------------------------------------------------

if arg[1] == 'demo' or arg[1] == 'predict-test' then
    -- Test set annotations do not have ground truth part locations, but provide
    -- information about the location and scale of people in each image.
    a = loadAnnotations('test')

elseif arg[1] == 'predict-valid' or arg[1] == 'eval' then
    -- Validation set annotations on the other hand, provide part locations,
    -- visibility information, normalization factors for final evaluation, etc.
    a = loadAnnotations('valid')

else
    print("Please use one of the following input arguments:")
    print("    demo - Generate and display results on a few demo images")
    print("    predict-valid - Generate predictions on the validation set (MPII images must be available in 'images' directory)")
    print("    predict-test - Generate predictions on the test set")
    print("    eval - Run basic evaluation on predictions from the validation set")
    return
end

m = torch.load('umich-stacked-hourglass.t7')   -- Load pre-trained model

if arg[1] == 'demo' then
    idxs = torch.Tensor({695, 3611, 2486, 7424, 10032, 5, 4829})
    -- If all the MPII images are available, use the following line to see a random sampling of images
    -- idxs = torch.randperm(a.nsamples):sub(1,10)
else
    idxs = torch.range(1,a.nsamples)
end

if arg[1] == 'eval' then
    nsamples = 0
else
    nsamples = idxs:nElement() 
    -- Displays a convenient progress bar
    xlua.progress(0,nsamples)
    preds = torch.Tensor(nsamples,16,2)
end

--------------------------------------------------------------------------------
-- Main loop
--------------------------------------------------------------------------------

function load_image(i)
    local im = image.load('images/' .. a['images'][idxs[i]])
    local center = a['center'][idxs[i]]
    local scale = a['scale'][idxs[i]]
    local inp = crop(im, center, scale, 0, 256)
    local info = {
        center = center,
        scale = scale
    }
    return inp:view(1,3,256,256), info
end

local minibatchSize = 8
print('loading tmp image')
local tmp_image = load_image(1)
print('loaded tmp image')
local dataBlock = tmp_image:zeros(minibatchSize,tmp_image:size(2),tmp_image:size(3),tmp_image:size(4))
print('initialized batch block')
tmp_image = nil

local startIdx = 0
function populate_block(startIdx)
    local num_in_batch = 0
    local batchInfo = {}
    for i = 1,minibatchSize do
        local idx  = startIdx + i
        if(idx > nsamples) then break end
        num_in_batch = num_in_batch + 1
        local im, info = load_image(idx)
        dataBlock[i]:copy(im)
        table.insert(batchInfo,info)
    end

    return dataBlock:narrow(1,1,num_in_batch), batchInfo
end

while(startIdx <= nsamples) do
   local block, info = populate_block(startIdx)
   print(num_in_batch)
   startIdx = startIdx + block:size(1)
    -- Get network output
    local cudaBlock  = block:cuda()
    local out = m:forward(cudaBlock)
    cutorch.synchronize()
    cudaBlock = nil
    local hm = out[2]:float()
    hm[hm:lt(0)] = 0

    -- Get predictions (hm and img refer to the coordinate space)
    for i = 1,block:size(1) do
        local preds_hm, preds_img = getPreds(hm:narrow(1,i,1), info[i].center, info[i].scale)
        preds[i]:copy(preds_img)
    end
    xlua.progress(startIdx,nsamples)

    collectgarbage()
end

-- Save predictions
if arg[1] == 'predict-valid' then
    local predFile = hdf5.open('preds/valid-example.h5', 'w')
    predFile:write('preds', preds)
    predFile:close()
elseif arg[1] == 'predict-test' then
    local predFile = hdf5.open('preds/test.h5', 'w')
    predFile:write('preds', preds)
    predFile:close()
elseif arg[1] == 'demo' then
    w.window:close()
end

--------------------------------------------------------------------------------
-- Evaluation code
--------------------------------------------------------------------------------

if arg[1] == 'eval' then
    -- Calculate distances given each set of predictions
    local labels = {'valid-example','valid-ours'}
    local dists = {}
    for i = 1,#labels do
        local predFile = hdf5.open('preds/' .. labels[i] .. '.h5','r')
        local preds = predFile:read('preds'):all()
        table.insert(dists,calcDists(preds, a.part, a.normalize))
    end

    require 'gnuplot'
    gnuplot.raw('set bmargin 1')
    gnuplot.raw('set lmargin 3.2')
    gnuplot.raw('set rmargin 2')    
    gnuplot.raw('set multiplot layout 2,3 title "MPII Validation Set Performance (PCKh)"')
    gnuplot.raw('set xtics font ",6"')
    gnuplot.raw('set ytics font ",6"')
    displayPCK(dists, {9,10}, labels, 'Head')
    displayPCK(dists, {2,5}, labels, 'Knee')
    displayPCK(dists, {1,6}, labels, 'Ankle')
    gnuplot.raw('set tmargin 2.5')
    gnuplot.raw('set bmargin 1.5')
    displayPCK(dists, {13,14}, labels, 'Shoulder')
    displayPCK(dists, {12,15}, labels, 'Elbow')
    displayPCK(dists, {11,16}, labels, 'Wrist', true)
    gnuplot.raw('unset multiplot')
end
