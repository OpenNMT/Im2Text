 --[[ Model, adapted from https://github.com/harvardnlp/seq2seq-attn/blob/master/train.lua
--]]
require 'nn'
require 'cudnn'
require 'optim'
require 'paths'
require 'src.cnn'

local model = torch.class('WYGIWYS')

-- constructor
function model:__init(optim)
  self.optim = optim
end

-- in test phase, open a file for predictions
function model:setOutputDirectory(outputDir)
  if not paths.dirp(outputDir) then
    paths.mkdir(outputDir)
  end
  local outputPath = paths.concat(outputDir, 'results.txt')
  local outputFile, err = io.open(outputPath, "w")
  if err then
    _G.logger:error('Output file %s cannot be created', outputPath)
    os.exit(1)
  end
  self.outputFile = outputFile
  self.outputPath = outputPath
end

-- load model from model_path
function model:load(modelPath, config)
  assert(paths.filep(modelPath), string.format('Model file %s does not exist!', modelPath))

  local checkpoint = torch.load(modelPath)
  local loadedModel, modelConfig = checkpoint[1], checkpoint[2]
  self.models = {}
  self.models.cnn = loadedModel[1]:double()
  self.models.encoder = onmt.BiEncoder.load(loadedModel[2])
  self.models.decoder = onmt.Decoder.load(loadedModel[3])
  self.models.posEmbeddingFw, self.models.posEmbeddingBw = loadedModel[4]:double(), loadedModel[5]:double()
  self.numSteps = checkpoint[3]
  self.optimState = checkpoint[4]
  _G.idToVocab = checkpoint[5] -- _G.idToVocab is global

  -- Load model structure parameters
  self.config = {}
  self.config.phase = config.phase
  self.config.cnnFeatureSize = modelConfig.cnnFeatureSize
  self.config.encoderNumHidden = modelConfig.encoderNumHidden
  self.config.encoderNumLayers = modelConfig.encoderNumLayers
  self.config.decoderNumHidden = self.config.encoderNumHidden * 2 -- the decoder rnn size is the same as the output size of biEncoder
  self.config.decoderNumLayers = modelConfig.decoderNumLayers
  self.config.targetVocabSize = #_G.idToVocab + 4
  self.config.targetEmbeddingSize = modelConfig.targetEmbeddingSize
  self.config.inputFeed = modelConfig.inputFeed

  self.config.maxEncoderLengthWidth = config.maxEncoderLengthWidth or modelConfig.maxEncoderLengthWidth
  self.config.maxEncoderLengthHeight = config.maxEncoderLengthHeight or modelConfig.maxEncoderLengthHeight
  self.config.maxDecoderLength = config.maxDecoderLength or modelConfig.maxDecoderLength
  self.config.batchSize = config.batch_size or modelConfig.batchSize

  -- If we want to allow higher images, since the trained positional embeddings are valid only up to modelConfig.maxEncoderLengthHeight, we use the largest available one to initialize the invalid embdddings
  if self.config.maxEncoderLengthHeight > modelConfig.maxEncoderLengthHeight then
  local posEmbeddingFw = nn.LookupTable(self.config.maxEncoderLengthHeight, self.config.encoderNumLayers * self.config.encoderNumHidden * 2)
  local posEmbeddingBw = nn.LookupTable(self.config.maxEncoderLengthHeight, self.config.encoderNumLayers * self.config.encoderNumHidden * 2)
  for i = 1, self.config.maxEncoderLengthHeight do
    local j = math.min(i, modelConfig.maxEncoderLengthHeight)
    posEmbeddingFw.weight[i] = self.models.posEmbeddingFw.weight[j]
    posEmbeddingBw.weight[i] = self.models.posEmbeddingBw.weight[j]
  end
  self.models.posEmbeddingFw, self.models.posEmbeddingBw = posEmbeddingFw, posEmbeddingBw
  end

  -- build model
  self:_build()
end

-- create model with fresh parameters
function model:create(config)
  -- set parameters
  self.config = {}
  self.config.phase = config.phase
  self.config.cnnFeatureSize = 512
  self.config.batchSize = config.batch_size
  self.config.inputFeed = config.input_feed
  self.config.encoderNumHidden = config.encoder_num_hidden
  self.config.encoderNumLayers = config.encoder_num_layers
  self.config.decoderNumHidden = config.encoder_num_hidden * 2
  self.config.decoderNumLayers = config.decoder_num_layers
  self.config.targetEmbeddingSize = config.target_embedding_size
  self.config.targetVocabSize = config.targetVocabSize
  self.config.maxEncoderLengthWidth = config.maxEncoderLengthWidth
  self.config.maxEncoderLengthHeight = config.maxEncoderLengthHeight
  self.config.maxDecoderLength = config.maxDecoderLength

  -- Create model modules
  self.models = {}
  -- positional embeddings
  self.models.posEmbeddingFw = nn.LookupTable(self.config.maxEncoderLengthHeight, self.config.encoderNumLayers * self.config.encoderNumHidden * 2)
  self.models.posEmbeddingBw = nn.LookupTable(self.config.maxEncoderLengthHeight, self.config.encoderNumLayers * self.config.encoderNumHidden * 2)
  -- CNN model, input size: (batchSize, 1, height, width), output size: (batchSize, sequenceLength, cnnFeatureSize)
  self.models.cnn = createCNNModel()
  -- biLSTM encoder
  --local encoderRnn = onmt.LSTM.new(self.config.encoderNumLayers, self.config.cnnFeatureSize, self.config.encoderNumHidden, 0.0)
  --self.models.encoder = onmt.BiEncoder.new(config, nn.Identity())
  local encoderConfig = {layers = self.config.encoderNumLayers, rnn_size = self.config.encoderNumHidden*2,
    brnn=true, brnn_merge = 'concat', dropout = 0, rnn_type = 'LSTM'}
  local encoderInputNetwork = nn.Identity()
  encoderInputNetwork.inputSize = self.config.cnnFeatureSize
  self.models.encoder = onmt.Factory.buildEncoder(encoderConfig, encoderInputNetwork, false)

  -- decoder
  local inputSize = self.config.targetEmbeddingSize
  if self.config.inputFeed then
    inputSize = inputSize + self.config.decoderNumHidden
  end
  local generator = onmt.Generator.new(self.config.decoderNumHidden, self.config.targetVocabSize)
  generator:cuda()
  local inputNetwork = onmt.WordEmbedding.new(self.config.targetVocabSize, self.config.targetEmbeddingSize)
  inputNetwork.inputSize = self.config.targetEmbeddingSize
  local attentionModel = onmt.GlobalAttention({global_attention='general'}, self.config.decoderNumHidden)
  local inputFeed = 0
  if self.config.inputFeed then
    inputFeed = 1
  end
  local decoderConfig = {input_feed = inputFeed, layers = self.config.decoderNumLayers, rnn_size = self.config.decoderNumHidden,
    residual = false, dropout = 0, dropout_input = false}
  self.models.decoder = onmt.Decoder.new(decoderConfig, inputNetwork, generator, attentionModel)

  self.numSteps = 0
  self._init = true

  self:_build()
end

-- build
function model:_build()

  -- log options
  for k, v in pairs(self.config) do
    _G.logger:info('%s: %s', k, v)
  end

  -- create criterion
  self.criterion = onmt.ParallelClassNLLCriterion({self.config.targetVocabSize})
  self.criterion:cuda()

  -- convert to cuda
  self.layers = {self.models.cnn, self.models.encoder, self.models.decoder, self.models.posEmbeddingFw, self.models.posEmbeddingBw}
  for i = 1, #self.layers do
    onmt.utils.Cuda.convert(self.layers[i])
  end
  onmt.utils.Cuda.convert(self.criterion)

  self.contextProto = onmt.utils.Cuda.convert(torch.zeros(self.config.batchSize, self.config.maxEncoderLengthWidth * self.config.maxEncoderLengthHeight, 2 * self.config.encoderNumHidden))
  self.models.cnnGradProto = onmt.utils.Cuda.convert(torch.zeros(self.config.maxEncoderLengthHeight, self.config.batchSize, self.config.maxEncoderLengthWidth, self.config.cnnFeatureSize))

  local numParams = 0
  self.params, self.gradParams = {}, {}
  for i = 1, #self.layers do
    local p, gp = self.layers[i]:getParameters()
    if self._init then
      p:uniform(-0.05,0.05)
    end
    numParams = numParams + p:size(1)
    self.params[i] = p
    self.gradParams[i] = gp
  end
  _G.logger:info('Number of parameters: %d', numParams)

  if self.config.phase == 'train' then
    self:_optimizeMemory()
  end
  collectgarbage()
end

-- one step forward (and optionally backward)
function model:step(inputBatch, isForwardOnly, beamSize)
  beamSize = beamSize or 1 -- default greedy decoding
  assert (beamSize <= self.config.targetVocabSize)
  local images = onmt.utils.Cuda.convert(inputBatch[1])
  local targetInput = onmt.utils.Cuda.convert(inputBatch[2])
  local targetOutput = onmt.utils.Cuda.convert(inputBatch[3])
  local numNonzeros = inputBatch[4]
  local imagePaths = inputBatch[5]

  local batchSize = images:size(1)
  local targetLength = targetInput:size(2)

  assert(targetLength <= self.config.maxDecoderLength, string.format('maxDecoderLength (%d) < targetLength (%d)!', self.config.maxDecoderLength, targetLength))
  -- if isForwardOnly, then re-generate the targetInput with maxDecoderLength for fair evaluation
  if isForwardOnly then
    local targetInputTemp = onmt.utils.Cuda.convert(torch.IntTensor(batchSize, self.config.maxDecoderLength)):fill(onmt.Constants.PAD)
    targetInputTemp[{{}, {1,targetLength}}]:copy(targetInput)
    targetInput = targetInputTemp
    local targetOutputTemp = onmt.utils.Cuda.convert(torch.IntTensor(batchSize, self.config.maxDecoderLength)):fill(onmt.Constants.PAD)
    targetOutputTemp[{{}, {1,targetLength}}]:copy(targetOutput)
    targetOutput = targetOutputTemp
    targetLength = self.config.maxDecoderLength
  end

  -- set phase
  if not isForwardOnly then
    self.models.cnn:training()
    self.models.encoder:training()
    self.models.decoder:training()
    self.models.posEmbeddingFw:training()
    self.models.posEmbeddingBw:training()
  else
    self.models.cnn:evaluate()
    self.models.encoder:evaluate()
    self.models.decoder:evaluate()
    self.models.posEmbeddingFw:evaluate()
    self.models.posEmbeddingBw:evaluate()
  end

  -- given parameters, evaluate loss (and optionally calculate gradients)
  local feval = function()
    local targetIn = targetInput:transpose(1,2)
    local targetOut = targetOutput:transpose(1,2)
    local cnnOutputs = self.models.cnn:forward(images) -- list of (batchSize, featureMapWidth, cnnFeatureSize)
    local featureMapHeight = #cnnOutputs
    local featureMapWidth = cnnOutputs[1]:size(2)
    local context = self.contextProto[{{1, batchSize}, {1, featureMapHeight * featureMapWidth}}]
    local decoderBatch = Batch():setTargetInput(targetIn):setTargetOutput(targetOut)
    decoderBatch.sourceLength = context:size(2)
    decoderBatch.sourceSize = onmt.utils.Cuda.convert(torch.IntTensor(batchSize)):fill(context:size(2))
    for i = 1, featureMapHeight do
      local pos = onmt.utils.Cuda.convert(torch.zeros(batchSize)):fill(i)
      local posEmbeddingFw  = self.models.posEmbeddingFw:forward(pos):view(1, batchSize, -1) -- (1, batchSize, cnnFeatureSize)
      local posEmbeddingBw  = self.models.posEmbeddingBw:forward(pos):view(1, batchSize, -1)
      local cnnOutput = cnnOutputs[i] -- (batchSize, featureMapWidth, cnnFeatureSize)
      local source = cnnOutput:transpose(1, 2) -- (featureMapWidth, batchSize, cnnFeatureSize)
      source = torch.cat(posEmbeddingFw, source, 1)
      source = torch.cat(source, posEmbeddingBw, 1)
      local encoderBatch = Batch():setSourceInput(source)
      local _, rowContext = self.models.encoder:forward(encoderBatch)
      for t = 1, featureMapWidth do
        local index = (i - 1) * featureMapWidth + t
        context[{{}, index, {}}]:copy(rowContext[{{}, t+1, {}}])
      end
    end

    -- evaluate loss (and optionally do backward)
    local loss, numCorrect
    numCorrect = 0
    if isForwardOnly then
      if self.outputFile then
        -- Specify how to go one step forward.
        local advancer = onmt.translate.DecoderAdvancer.new(self.models.decoder, decoderBatch, context, self.config.maxDecoderLength)
        
        -- Conduct beam search.
        local beamSearcher = onmt.translate.BeamSearcher.new(advancer)
        local results = beamSearcher:search(beamSize, 1)
        local predTarget = onmt.utils.Cuda.convert(torch.zeros(batchSize, targetLength)):fill(onmt.Constants.PAD)
        for b = 1, batchSize do
          local tokens = results[b][1].tokens
          for t = 1, #tokens do
            predTarget[b][t] = tokens[t]
          end
        end
        local predLabels = targetsTensorToLabelStrings(predTarget)
        local goldLabels = targetsTensorToLabelStrings(targetOutput)
        local editDistanceRate = evalEditDistanceRate(goldLabels, predLabels)
        numCorrect = batchSize - editDistanceRate
        for i = 1, #imagePaths do
          _G.logger:info('%s\t%s\n', imagePaths[i], predLabels[i])
          self.outputFile:write(string.format('%s\t%s\n', imagePaths[i], predLabels[i]))
        end
        self.outputFile:flush()
      end
      -- get loss
      self.models.decoder:maskPadding()
      loss = self.models.decoder:computeLoss(decoderBatch, nil, context, self.criterion) / batchSize
    else -- isForwardOnly == false
      local decoderOutputs = self.models.decoder:forward(decoderBatch, nil, context)
      local _, gradContext, totalLoss = self.models.decoder:backward(decoderBatch, decoderOutputs, self.criterion)
      loss = totalLoss / batchSize
      gradContext = gradContext:contiguous():view(batchSize, featureMapHeight, featureMapWidth, -1) -- (batchSize, featureMapHeight, featureMapWidth, cnnFeatureSize)
      local gradPadding = onmt.utils.Cuda.convert(torch.zeros(batchSize, featureMapHeight, 1, self.config.cnnFeatureSize))
      gradContext = torch.cat(gradPadding, gradContext, 3)
      gradContext = torch.cat(gradContext, gradPadding, 3)
      local cnnGrad = self.models.cnnGradProto[{ {1, featureMapHeight}, {1, batchSize}, {1, featureMapWidth}, {} }]
      for i = 1, featureMapHeight do
        local cnnOutput = cnnOutputs[i]
        local source = cnnOutput:transpose(1,2)
        local pos = onmt.utils.Cuda.convert(torch.zeros(batchSize)):fill(i)
        local posEmbeddingFw = self.models.posEmbeddingFw:forward(pos):view(1, batchSize, -1)
        local posEmbeddingBw = self.models.posEmbeddingBw:forward(pos):view(1, batchSize, -1)
        source = torch.cat(posEmbeddingFw, source, 1)
        source = torch.cat(source, posEmbeddingBw, 1) -- (featureMapWidth + 2, batchSize, cnnFeatureSize)
        local encoderBatch = Batch():setSourceInput(source)
        self.models.encoder:forward(encoderBatch)
        local rowContextGrad = self.models.encoder:backward(encoderBatch, nil, gradContext:select(2,i))
        for t = 1, featureMapWidth do
          cnnGrad[{i, {}, t, {}}]:copy(rowContextGrad[t + 1])
        end
        self.models.posEmbeddingFw:backward(pos, rowContextGrad[1])
        self.models.posEmbeddingBw:backward(pos, rowContextGrad[featureMapWidth + 2])
      end
      -- cnn
      cnnGrad = cnnGrad:split(1, 1)
      for i = 1, #cnnGrad do
        cnnGrad[i] = cnnGrad[i]:contiguous():view(batchSize, featureMapWidth, -1)
      end
      self.models.cnn:backward(images, cnnGrad)
      collectgarbage()
    end
    return loss, self.gradParams, {numNonzeros, numCorrect}
  end
  if not isForwardOnly then
    -- optimizer
    self.optim:zeroGrad(self.gradParams)
    local loss, _, stats = feval(self.params)
    self.optim:prepareGrad(self.gradParams, 20.0)
    self.optim:updateParams(self.params, self.gradParams)

    return loss * batchSize, stats
  else
    local loss, _, stats = feval(self.params)
    return loss * batchSize, stats
  end
end

-- Optimize Memory Usage by sharing output and gradInput among clones
function model:_optimizeMemory()
  self.models.encoder:training()
  self.models.decoder:training()
  _G.logger:info('Preparing memory optimization...')
  local memoryOptimizer = onmt.utils.MemoryOptimizer.new({self.models.encoder, self.models.decoder})

  -- Initialize all intermediate tensors with a first batch.
  local source = onmt.utils.Cuda.convert(torch.zeros(1, 1, self.config.cnnFeatureSize))
  local targetIn = onmt.utils.Cuda.convert(torch.ones(1, 1))
  local targetOut = targetIn:clone()
  local batch = Batch():setSourceInput(source):setTargetInput(targetIn):setTargetOutput(targetOut)
  self.models.encoder:forward(batch)
  local context = onmt.utils.Cuda.convert(torch.zeros(1, 1, 2 * self.config.encoderNumHidden))
  local decOutputs = self.models.decoder:forward(batch, nil, context)
  decOutputs = onmt.utils.Tensor.recursiveClone(decOutputs)
  local _, gradContext = self.models.decoder:backward(batch, decOutputs, self.criterion)
  self.models.encoder:backward(batch, nil, gradContext)

  local sharedSize, totSize = memoryOptimizer:optimize()
  _G.logger:info(' * sharing %d%% of output/gradInput tensors memory between clones', (sharedSize / totSize)*100)
end

-- Save model to model_path
function model:save(modelPath)
  for i = 1, #self.layers do
    self.layers[i]:clearState()
  end
  torch.save(modelPath, {{self.models.cnn, self.models.encoder:serialize(), self.models.decoder:serialize(), self.models.posEmbeddingFw, self.models.posEmbeddingBw}, self.config, self.numSteps, self.optimState, _G.idToVocab})
end

-- destructor
function model:shutDown()
  if self.outputFile then
    self.outputFile:close()
    _G.logger:info('Results saved to %s.', self.outputPath)
  end
end
