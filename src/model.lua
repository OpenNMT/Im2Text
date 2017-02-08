--[[ WYGIWYS Model. ]]
require 'onmt.models.Model'
local model, parent = torch.class('onmt.Models.wygiwys', 'onmt.Model')

local model_options = {
  {'-encoder_num_hidden', 256, [[Number of hidden units in encoder cell]]},
  {'-encoder_num_layers', 1, [[Number of hidden layers in encoder cell]]},
  {'-decoder_num_layers', 1, [[Number of hidden units in decoder cell]]},
  {'-target_embedding_size', 80, [[Embedding dimension for each target]]},
  {'-input_feed', false, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]]},
  {'-brnn_merge', 'sum', [[Merge action for the bidirectional hidden states: concat or sum]]}
}

function model.declareOpts(cmd)
  cmd:setCmdLineOptions(model_options, "WYGIWYS Model")
end


local function createCNN(useCudnn)
  --[[ See arXiv: 1609.04938v1 
      Conv                                    Pool
      c:512, k:(3,3), s:(1,1), p:(0,0), bn    -
      c:512, k:(3,3), s:(1,1), p:(1,1), bn    po:(2,1), s:(2,1), p:(0,0)
      c:256, k:(3,3), s:(1,1), p:(1,1)        po:(1,2), s:(1,2), p(0,0)
      c:256, k:(3,3), s:(1,1), p:(1,1), bn    -
      c:128, k:(3,3), s:(1,1), p:(1,1)        po:(2,2), s:(2,2), p:(0,0)
      c:64, k:(3,3), s:(1,1), p:(1,1)         po:(2,2), s:(2,2), p(0,0)
  
      Table 2: CNN  specification.  ‘Conv‘:  convolution  layer,  ‘Pool:
      max-pooling layer. ‘c’: number of filters, ‘k’: kernel size, ‘s’: stride
      size, ‘p’: padding size, ‘po’: , ‘bn’: with batch normalization. The
      sizes are in order (height, width).
  --]]
  local cnn = nn.Sequential()

  -- Format: Conv, Batch Normalization, Pool
  local cnnSpecs = {
    {{64, 3, 3, 1, 1, 1, 1},  false, {2, 2, 2, 2, 0, 0}},
    {{128, 3, 3, 1, 1, 1, 1}, false, {2, 2, 2, 2, 0, 0}},
    {{256, 3, 3, 1, 1, 1, 1}, true},
    {{256, 3, 3, 1, 1, 1, 1}, false, {1, 2, 1, 2, 0, 0}},
    {{512, 3, 3, 1, 1, 1, 1}, true, {2, 1, 2, 1, 0, 0}},
    {{512, 3, 3, 1, 1, 1, 1}, true}
  }

  -- input shape: (batchSize, 1, imgH, imgW)
  cnn:add(nn.AddConstant(-128.0))
  cnn:add(nn.MulConstant(1.0 / 128))

  for i = 1, #cnnSpecs do
    local channels
    if i == 1 then
      channels = 1
    else
      channels = cnnSpecs[i - 1][1][1]
    end
    -- Conv
    cnn:add(nn.SpatialConvolution(channels, table.unpack(cnnSpecs[i][1])))
    cnn.channels = cnnSpecs[i][1][1]
    -- Batch Normalization
    if cnnSpecs[i][2] then
      cnn:add(nn.SpatialBatchNormalization(cnnSpecs[i][1][1]))
    end
    -- RELU
    cnn:add(nn.ReLU(true))
    -- Pool
    if cnnSpecs[i][3] then
      cnn:add(nn.SpatialMaxPooling(table.unpack(cnnSpecs[i][3])))
    end
  end
  -- shape: (batchSize, channels, H, W) to (batchSize, H, W, channels)
  cnn:add(nn.Transpose({2, 3}, {3,4}))
  -- #H list of (batch_size, W, channels)
  cnn:add(nn.SplitTable(1, 3))

  if useCudnn then
    cudnn.convert(cnn, cudnn)
  end

  return cnn
end

-- If we want to allow higher images, since the trained positional embeddings are valid only up to previously seen max value, we use the largest available one to initialize the invalid embdddings
function model:_expandLookupTable(imgH)
  local validImgH = self.models.posFw.weight:size(1)
  if imgH > validImgH then
    local pFw, gpFw = self.models.posFw:getParameters()
    local pBw, gpBw = self.models.posBw:getParameters()
    local posFw = nn.LookupTable(imgH, self.args.encoder_num_layers * self.args.encoder_num_hidden * 2)
    local posBw = nn.LookupTable(imgH, self.args.encoder_num_layers * self.args.encoder_num_hidden * 2)
    posFw = onmt.utils.Cuda.convert(posFw)
    posBw = onmt.utils.Cuda.convert(posBw)
    local pFwNew, gpFwNew = posFw:getParameters()
    local pBwNew, gpBwNew = posBw:getParameters()
    for i = 1, imgH do
      local j = math.min(i, validImgH)
      posFw.weight[i]:copy(self.models.posFw.weight[j])
      posBw.weight[i]:copy(self.models.posBw.weight[j])
    end
    pFw:set(pFwNew)
    gpFw:set(gpFwNew)
    pBw:set(pBwNew)
    gpBw:set(gpBwNew)
    self.models.posFw, self.models.posBw = posFw, posBw
  end
end

function model:__init(args, datasetOrCheckpoint, verboseOrReplica)
  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.ExtendedCmdLine.getModuleOpts(args, model_options))
  -- Log options.
  for k, v in pairs(self.args) do
    _G.logger:info('%s: %s', k, v)
  end

  if type(datasetOrCheckpoint)=='Checkpoint' then
    local checkpoint = datasetOrCheckpoint
    local replica = verboseOrReplica
    self.models.cnn = onmt.Models.cnn
    self.models.encoder = onmt.Models.loadEncoder(checkpoint.models, replica)
    self.models.decoder = onmt.Models.loadDecoder(checkpoint.models, replica)
    self.models.posFw, self.models.posBw = checkpoint.models.posFw, checkpoint.models.posBw
  else
    local dataset = datasetOrCheckpoint
    local verbose = verboseOrReplica
    self.models = {}
    -- Create CNN.
    self.models.cnn = createCNN(onmt.utils.Cuda.activated)
    -- Create positional embeddings.
    self.models.posFw = nn.LookupTable(1, self.args.encoder_num_layers * self.args.encoder_num_hidden * 2)
    self.models.posBw = nn.LookupTable(1, self.args.encoder_num_layers * self.args.encoder_num_hidden * 2)
    -- Create biLSTM encoder.
    local encoderRnn = onmt.LSTM.new(self.args.encoder_num_layers, self.models.cnn.channels, self.args.encoder_num_hidden, 0.0)
    self.models.encoder = onmt.BiEncoder.new(nn.Identity(), encoderRnn, self.args.brnn_merge)
    -- Create decoder.
    local decoderNumHidden
    if self.args.brnn_merge == 'concat' then
      decoderNumHidden = self.args.encoder_num_hidden * 2
    else
      decoderNumHidden = self.args.encoder_num_hidden
    end
    local inputSize = self.args.target_embedding_size
    if self.args.input_feed then
      inputSize = inputSize + decoderNumHidden
    end
    local decoderRnn = onmt.LSTM.new(self.args.decoder_num_layers, inputSize, decoderNumHidden, 0.0)
    local generator = onmt.Generator.new(decoderNumHidden, dataset.dicts:size())
    local inputNetwork = onmt.WordEmbedding.new(dataset.dicts:size(), self.args.target_embedding_size)
    self.models.decoder = onmt.Decoder.new(inputNetwork, decoderRnn, generator, self.args.input_feed)
  end
end

-- Returns model name.
function model.modelName()
  return "WYSIYWG"
end

-- Returns expected dataMode.
function model.dataType()
  return "IMG-TEXT"
end

function model:forwardComputeLoss(batch, criterion)
  local loss = 0

  local images = batch.sourceInput:transpose(4,3):transpose(3,2):transpose(2,1)
  images = onmt.utils.Cuda.convert(images)
  local batchSize = images:size(1)

  local cnnOutputs = self.models.cnn:forward(images) -- list of (batchSize, featureMapWidth, cnnFeatureSize)
  if doProfile then _G.profiler:stop("cnn.fwd") end
  local featureMapHeight = #cnnOutputs
  self:_expandLookupTable(featureMapHeight)
  local featureMapWidth = cnnOutputs[1]:size(2)
  local decoderNumHidden
  if self.args.brnn_merge == 'concat' then
    decoderNumHidden = self.args.encoder_num_hidden * 2
  else
    decoderNumHidden = self.args.encoder_num_hidden
  end
  local context = torch.Tensor(batchSize, featureMapHeight * featureMapWidth, decoderNumHidden)
  context = onmt.utils.Cuda.convert(context)
  batch.sourceLength = context:size(2)
  batch.sourceSize = onmt.utils.Cuda.convert(torch.IntTensor(batchSize)):fill(context:size(2))
  for i = 1, featureMapHeight do
    local pos = onmt.utils.Cuda.convert(torch.zeros(batchSize)):fill(i)
    local posFw  = self.models.posFw:forward(pos):view(1, batchSize, -1) -- (1, batchSize, cnnFeatureSize)
    local posBw  = self.models.posBw:forward(pos):view(1, batchSize, -1)
    local cnnOutput = cnnOutputs[i] -- (batchSize, featureMapWidth, cnnFeatureSize)
    local source = cnnOutput:transpose(1, 2) -- (featureMapWidth, batchSize, cnnFeatureSize)
    source = torch.cat(posFw, source, 1)
    source = torch.cat(source, posBw, 1)
    local encoderBatch = Batch():setSourceInput(source)
    local _, rowContext = self.models.encoder:forward(encoderBatch)
    for t = 1, featureMapWidth do
      local index = (i - 1) * featureMapWidth + t
      context[{{}, index, {}}]:copy(rowContext[{{}, t+1, {}}])
    end
  end

  self.models.decoder:maskPadding()
  loss = self.models.decoder:computeLoss(batch, nil, context, criterion)
  return loss
end

function model:buildCriterion(dataset)
  return onmt.Criterion.new(dataset.dicts:size(), {})
end

function model:countTokens(batch)
  return batch.targetNonZeros
end

function model:trainNetwork(batch, criterion, doProfile)
  local loss = 0

  local images = batch.sourceInput:transpose(4,3):transpose(3,2):transpose(2,1)
  images = onmt.utils.Cuda.convert(images)
  local batchSize = images:size(1)

  if doProfile then _G.profiler:start("cnn.fwd") end
  local cnnOutputs = self.models.cnn:forward(images) -- list of (batchSize, featureMapWidth, cnnFeatureSize)
  if doProfile then _G.profiler:stop("cnn.fwd") end
  local featureMapHeight = #cnnOutputs
  self:_expandLookupTable(featureMapHeight)
  local featureMapWidth = cnnOutputs[1]:size(2)
  local decoderNumHidden
  if self.args.brnn_merge == 'concat' then
    decoderNumHidden = self.args.encoder_num_hidden * 2
  else
    decoderNumHidden = self.args.encoder_num_hidden
  end
  local context = torch.Tensor(batchSize, featureMapHeight * featureMapWidth, decoderNumHidden)
  context = onmt.utils.Cuda.convert(context)
  batch.sourceLength = context:size(2)
  batch.sourceSize = onmt.utils.Cuda.convert(torch.IntTensor(batchSize)):fill(context:size(2))
  for i = 1, featureMapHeight do
    local pos = onmt.utils.Cuda.convert(torch.zeros(batchSize)):fill(i)
    if doProfile then _G.profiler:start("posFw.fwd") end
    local posFw  = self.models.posFw:forward(pos):view(1, batchSize, -1) -- (1, batchSize, cnnFeatureSize)
    if doProfile then _G.profiler:stop("posFw.fwd") end
    if doProfile then _G.profiler:start("posBw.fwd") end
    local posBw  = self.models.posBw:forward(pos):view(1, batchSize, -1)
    if doProfile then _G.profiler:stop("posBw.fwd") end
    local cnnOutput = cnnOutputs[i] -- (batchSize, featureMapWidth, cnnFeatureSize)
    local source = cnnOutput:transpose(1, 2) -- (featureMapWidth, batchSize, cnnFeatureSize)
    source = torch.cat(posFw, source, 1)
    source = torch.cat(source, posBw, 1)
    local encoderBatch = Batch():setSourceInput(source)
    if doProfile then _G.profiler:start("encoder.fwd") end
    local _, rowContext = self.models.encoder:forward(encoderBatch)
    if doProfile then _G.profiler:stop("encoder.fwd") end
    for t = 1, featureMapWidth do
      local index = (i - 1) * featureMapWidth + t
      context[{{}, index, {}}]:copy(rowContext[{{}, t+1, {}}])
    end
  end

  if doProfile then _G.profiler:start("decoder.fwd") end
  local decoderOutputs = self.models.decoder:forward(batch, nil, context)
  if doProfile then _G.profiler:stop("decoder.fwd") end
  if doProfile then _G.profiler:start("decoder.bwd") end
  local _, gradContext, loss = self.models.decoder:backward(batch, decoderOutputs, criterion)
  if doProfile then _G.profiler:stop("decoder.bwd") end
  gradContext = gradContext:contiguous():view(batchSize, featureMapHeight, featureMapWidth, -1) -- (batchSize, featureMapHeight, featureMapWidth, cnnFeatureSize)
  local gradPadding = onmt.utils.Cuda.convert(torch.zeros(batchSize, featureMapHeight, 1, self.models.cnn.channels))
  gradContext = torch.cat(gradPadding, gradContext, 3)
  gradContext = torch.cat(gradContext, gradPadding, 3)
  local cnnGrad = torch.Tensor(featureMapHeight, batchSize, featureMapWidth, self.models.cnn.channels)
  cnnGrad = onmt.utils.Cuda.convert(cnnGrad):zero()
  for i = 1, featureMapHeight do
    local cnnOutput = cnnOutputs[i]
    local source = cnnOutput:transpose(1,2)
    local pos = onmt.utils.Cuda.convert(torch.zeros(batchSize)):fill(i)
    if doProfile then _G.profiler:start("posFw.fwd2") end
    local posFw = self.models.posFw:forward(pos):view(1, batchSize, -1)
    if doProfile then _G.profiler:stop("posFw.fwd2") end
    if doProfile then _G.profiler:start("posBw.fwd2") end
    local posBw = self.models.posBw:forward(pos):view(1, batchSize, -1)
    if doProfile then _G.profiler:stop("posBw.fwd2") end
    source = torch.cat(posFw, source, 1)
    source = torch.cat(source, posBw, 1) -- (featureMapWidth + 2, batchSize, cnnFeatureSize)
    local encoderBatch = Batch():setSourceInput(source)
    if doProfile then _G.profiler:start("encoder.fwd2") end
    self.models.encoder:forward(encoderBatch)
    if doProfile then _G.profiler:stop("encoder.fwd2") end
    if doProfile then _G.profiler:start("encoder.bwd") end
    local rowContextGrad = self.models.encoder:backward(encoderBatch, nil, gradContext:select(2,i))
    if doProfile then _G.profiler:stop("encoder.bwd") end
    for t = 1, featureMapWidth do
      cnnGrad[{i, {}, t, {}}]:copy(rowContextGrad[t + 1])
    end
    if doProfile then _G.profiler:start("posFw.bwd") end
    self.models.posFw:backward(pos, rowContextGrad[1])
    if doProfile then _G.profiler:stop("posFw.bwd") end
    if doProfile then _G.profiler:start("posBw.bwd") end
    self.models.posBw:backward(pos, rowContextGrad[featureMapWidth + 2])
    if doProfile then _G.profiler:stop("posBw.bwd") end
  end
  -- cnn
  cnnGrad = cnnGrad:split(1, 1)
  for i = 1, #cnnGrad do
    cnnGrad[i] = cnnGrad[i]:contiguous():view(batchSize, featureMapWidth, -1)
  end
  if doProfile then _G.profiler:start("cnn.bwd") end
  self.models.cnn:backward(images, cnnGrad)
  if doProfile then _G.profiler:stop("cnn.bwd") end

  collectgarbage()

  return loss
end

return model
