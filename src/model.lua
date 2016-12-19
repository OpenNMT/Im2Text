 --[[ Model, adapted from https://github.com/harvardnlp/seq2seq-attn/blob/master/train.lua
--]]
require 'nn'
require 'cudnn'
require 'optim'
require 'paths'
require 'src.cnn'

local model = torch.class('Model')

-- constructor
function model:__init()
end

-- in test phase, open a file for predictions
function model:setOutputDirectory(outputDir)
  if not paths.dirp(outputDir) then
    paths.mkdir(outputDir)
  end
  local outputPath = paths.concat(outputDir, 'results.txt')
  local outputFile, err = io.open(outputPath, "w")
  if err then 
    logger:error('Output file %s cannot be created', outputPath)
    os.exit(1)
  end
  self.outputFile = outputFile
  self.outputPath = outputPath
end

-- load model from model_path
function model:load(modelPath, config)
  assert(paths.filep(modelPath), string.format('Model file %s does not exist!', modelPath))

  local checkpoint = torch.load(modelPath)
  local model, modelConfig = checkpoint[1], checkpoint[2]
  self.cnn = model[1]:double()
  self.encoder = onmt.BiEncoder.load(model[2])
  self.decoder = onmt.Decoder.load(model[3])
  self.posEmbeddingFw, self.posEmbeddingBw = model[4]:double(), model[5]:double()
  self.numSteps = checkpoint[3]
  self.optimState = checkpoint[4]
  idToVocab = checkpoint[5] -- idToVocab is global

  -- Load model structure parameters
  self.config = {}
  self.config.cnnFeatureSize = modelConfig.cnnFeatureSize
  self.config.encoderNumHidden = modelConfig.encoderNumHidden
  self.config.encoderNumLayers = modelConfig.encoderNumLayers
  self.config.decoderNumHidden = self.config.encoderNumHidden * 2 -- the decoder rnn size is the same as the output size of biEncoder
  self.config.decoderNumLayers = modelConfig.decoderNumLayers
  self.config.targetVocabSize = #idToVocab + 4
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
    posEmbeddingFw.weight[i] = self.posEmbeddingFw.weight[j]
    posEmbeddingBw.weight[i] = self.posEmbeddingBw.weight[j]
  end
  self.posEmbeddingFw, self.posEmbeddingBw = posEmbeddingFw, posEmbeddingBw
  end

  -- build model
  self:_build()
end

-- create model with fresh parameters
function model:create(config)
  -- set parameters
  self.config = {}
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
  -- positional embeddings
  self.posEmbeddingFw = nn.LookupTable(self.config.maxEncoderLengthHeight, self.config.encoderNumLayers * self.config.encoderNumHidden * 2)
  self.posEmbeddingBw = nn.LookupTable(self.config.maxEncoderLengthHeight, self.config.encoderNumLayers * self.config.encoderNumHidden * 2)
  -- CNN model, input size: (batchSize, 1, height, width), output size: (batchSize, sequenceLength, cnnFeatureSize)
  self.cnn = createCNNModel()
  -- biLSTM encoder
  local encoderRnn = onmt.LSTM.new(self.config.encoderNumLayers, self.config.cnnFeatureSize, self.config.encoderNumHidden, 0.0)
  self.encoder = onmt.BiEncoder.new(nn.Identity(), encoderRnn, 'concat')

  -- decoder
  local inputSize = self.config.targetEmbeddingSize
  if self.config.inputFeed then
    inputSize = inputSize + self.config.decoderNumHidden
  end
  local decoderRnn = onmt.LSTM.new(self.config.decoderNumLayers, inputSize, self.config.decoderNumHidden, 0.0)
  local generator = onmt.Generator.new(self.config.decoderNumHidden, self.config.targetVocabSize)
  local inputNetwork = onmt.WordEmbedding.new(self.config.targetVocabSize, self.config.targetEmbeddingSize)
  self.decoder = onmt.Decoder.new(inputNetwork, decoderRnn, generator, self.config.inputFeed)

  self.numSteps = 0
  self._init = true

  self.optimState = {}
  self.optimState.learningRate = config.learningRate
  self.optimState.method = 'sgd'

  self:_build()
end

-- build
function model:_build()

  -- log options
  for k, v in pairs(self.config) do
    logger:info('%s: %s', k, v)
  end

  -- create criterion
  self.criterion = nn.ParallelCriterion(false)
  local weights = torch.ones(self.config.targetVocabSize)
  weights[1] = 0
  local nll = nn.ClassNLLCriterion(weights)
  nll.sizeAverage = false
  self.criterion:add(nll)

  -- convert to cuda
  self.layers = {self.cnn, self.encoder, self.decoder, self.posEmbeddingFw, self.posEmbeddingBw}
  for i = 1, #self.layers do
    onmt.utils.Cuda.convert(self.layers[i])
  end
  onmt.utils.Cuda.convert(self.criterion)

  self.contextProto = onmt.utils.Cuda.convert(torch.zeros(self.config.batchSize, self.config.maxEncoderLengthWidth * self.config.maxEncoderLengthHeight, 2 * self.config.encoderNumHidden))
  self.cnnGradProto = onmt.utils.Cuda.convert(torch.zeros(self.config.maxEncoderLengthHeight, self.config.batchSize, self.config.maxEncoderLengthWidth, self.config.cnnFeatureSize))

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
  logger:info('Number of parameters: %d', numParams)


  collectgarbage()
end

-- one step forward (and optionally backward)
function model:step(inputBatch, isForwardOnly, beamSize)
  if isForwardOnly then
    beamSize = beamSize or 1 -- default greedy decoding
    beamSize = math.min(beamSize, self.config.targetVocabSize)
    if not self.initBeam then
      self.initBeam = true
      local beamDecoderInitState = onmt.utils.Cuda.convert(torch.zeros(self.config.batchSize*beamSize, self.config.decoderNumHidden))
      self.beamScores = onmt.utils.Cuda.convert(torch.zeros(self.config.batchSize, beamSize))
      self.currentIndicesHistory = {}
      self.beamParentsHistory = {}
    else
      self.beamScores:zero()
      self.currentIndicesHistory = {}
      self.beamParentsHistory = {}
    end
  else -- isForwardOnly == false
    if self.initBeam then
      self.initBeam = false
      self.currentIndicesHistory = {}
      self.beamParentsHistory = {}
      self.beamScores = nil
      collectgarbage()
    end
  end
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
    self.cnn:training()
    self.encoder:training()
    self.decoder:training()
    self.posEmbeddingFw:training()
    self.posEmbeddingBw:training()
  else
    self.cnn:evaluate()
    self.encoder:evaluate()
    self.decoder:evaluate()
    self.posEmbeddingFw:evaluate()
    self.posEmbeddingBw:evaluate()
  end

  -- given parameters, evaluate loss (and optionally calculate gradients)
  local feval = function(p)
    targetIn = targetInput:transpose(1,2)
    targetOut = targetOutput:transpose(1,2)
    local cnnOutputs = self.cnn:forward(images) -- list of (batchSize, featureMapWidth, cnnFeatureSize)
    local counter = 1
    local featureMapHeight = #cnnOutputs
    local featureMapWidth = cnnOutputs[1]:size(2)
    local context = self.contextProto[{{1, batchSize}, {1, featureMapHeight * featureMapWidth}}]
    local decoderBatch = Batch():setTargetInput(targetIn):setTargetOutput(targetOut)
    decoderBatch.sourceLength = context:size(2)
    for i = 1, featureMapHeight do
      local pos = onmt.utils.Cuda.convert(torch.zeros(batchSize)):fill(i)
      local posEmbeddingFw  = self.posEmbeddingFw:forward(pos):view(1, batchSize, -1) -- (1, batchSize, cnnFeatureSize)
      local posEmbeddingBw  = self.posEmbeddingBw:forward(pos):view(1, batchSize, -1)
      local cnnOutput = cnnOutputs[i] -- (batchSize, featureMapWidth, cnnFeatureSize)
      local source = cnnOutput:transpose(1, 2) -- (featureMapWidth, batchSize, cnnFeatureSize)
      source = torch.cat(posEmbeddingFw, source, 1)
      source = torch.cat(source, posEmbeddingBw, 1)
      local encoderBatch = Batch():setSourceInput(source)
      local _, rowContext = self.encoder:forward(encoderBatch)
      for t = 1, featureMapWidth do
        index = (i - 1) * featureMapHeight + t
        context[{{}, index, {}}]:copy(rowContext[{{}, t+1, {}}])
      end
    end
    local decoderOutputs
    -- beam search
    if isForwardOnly then
      local beamReplicate = function(h)
        assert(1 <= h:dim() and h:dim() <= 3, 'does not support ndim except for 1, 2 and 3')
        local batchSize = h:size(1)
        if h:dim() == 1 then
          return h:contiguous():view(batchSize, 1):expand(batchSize, beamSize):contiguous():view(-1)
        elseif h:dim() == 2 then
          local size = h:size(2)
          return h:contiguous():view(batchSize, 1, size):expand(batchSize, beamSize, size):contiguous():view(batchSize * beamSize, size)
        else -- h:dim() == 3
          local size1, size2 = h:size(2), h:size(3)
          return h:contiguous():view(batchSize, 1, size1, size2):expand(batchSize, beamSize, size1, size2):contiguous():view(batchSize * beamSize, size1, size2)
        end
      end
      local beamContext = beamReplicate(context)
      local decoderStates = onmt.utils.Tensor.initTensorTable(self.decoder.args.numEffectiveLayers,
                onmt.utils.Cuda.convert(torch.Tensor()),
                { batchSize, self.decoder.args.rnnSize })
      local decoderInput, decoderOutput
      for t = 1, targetLength do
        local decoderContext
        if t == 1 then
          decoderInput = onmt.utils.Cuda.convert(torch.zeros(batchSize)):fill(onmt.Constants.BOS)
          decoderContext = context
        else
          decoderContext = beamContext
        end
        decoderOutput, decoderStates = self.decoder:forwardOne(decoderInput, decoderStates, decoderContext, decoderOutput, t)
        local probs = self.decoder.generator:forward(decoderOutput)[1] -- t ~= 1, (batchSize * beamSize, targetVocabSize); t == 1, (batchSize, targetVocabSize)
        local currentIndices, rawIndices
        local beamParents
        if t == 1 then
          self.beamScores, rawIndices = probs:topk(beamSize, true)
          rawIndices = onmt.utils.Cuda.convert(rawIndices:double())
          currentIndices = rawIndices
        else
          probs:select(2, onmt.Constants.PAD):maskedFill(decoderInput:eq(onmt.Constants.PAD), 0) -- once padding or EOS encountered, stuck at that point
          probs:select(2, onmt.Constants.PAD):maskedFill(decoderInput:eq(onmt.Constants.EOS), 0)
          local totalScores = (probs:view(batchSize, beamSize, self.config.targetVocabSize) + self.beamScores[{{1, batchSize}, {}}]:view(batchSize, beamSize, 1):expand(batchSize, beamSize, self.config.targetVocabSize)):view(batchSize, beamSize * self.config.targetVocabSize) -- (batchSize, beamSize * targetVocabSize)
          self.beamScores, rawIndices = totalScores:topk(beamSize, true) -- (batchSize, beamSize)
          rawIndices = onmt.utils.Cuda.convert(rawIndices:double())
          rawIndices:add(-1)
          currentIndices = onmt.utils.Cuda.convert(rawIndices:double():fmod(self.config.targetVocabSize)) + 1 -- (batchSize, beamSize)
        end
        beamParents = onmt.utils.Cuda.convert(rawIndices:int()/self.config.targetVocabSize + 1) -- (batchSize, beamSize)
        decoderInput = currentIndices:view(batchSize * beamSize)
        table.insert(self.currentIndicesHistory, currentIndices:clone())
        table.insert(self.beamParentsHistory, beamParents:clone())

        if self.config.inputFeed then
          if t == 1 then
            decoderOutput = beamReplicate(decoderOutput)
          end
          decoderOutput = decoderOutput:index(1, beamParents:view(-1) + onmt.utils.Cuda.convert(torch.range(0, (batchSize - 1) * beamSize, beamSize):long()):contiguous():view(batchSize, 1):expand(batchSize, beamSize):contiguous():view(-1))
        end
        for j = 1, #decoderStates do
          local decoderState = decoderStates[j] -- (batchSize * beamSize, decoderNumHidden)
          if t == 1 then
            decoderState = beamReplicate(decoderState)
          end
          decoderStates[j] = decoderState:index(1, beamParents:view(-1) + onmt.utils.Cuda.convert(torch.range(0, (batchSize - 1) * beamSize, beamSize):long()):contiguous():view(batchSize,1):expand(batchSize, beamSize):contiguous():view(-1))
        end
      end
    else -- isForwardOnly == false
      decoderOutputs = self.decoder:forward(decoderBatch, nil, context)
    end

    -- evaluate loss (and optionally do backward)
    local loss, numCorrect = 0.0, 0.0
    if isForwardOnly then
      -- final decoding
      local predTarget = onmt.utils.Cuda.convert(torch.zeros(batchSize, targetLength)):fill(onmt.Constants.PAD)
      local predScores, indices = torch.max(self.beamScores[{{1, batchSize},{}}], 2) -- (batchSize, 1)
      indices = onmt.utils.Cuda.convert(indices:double())
      predScores = predScores:view(-1) -- batchSize
      indices = indices:view(-1) -- batchSize
      local currentIndices = self.currentIndicesHistory[#self.currentIndicesHistory]:view(-1):index(1, indices + onmt.utils.Cuda.convert(torch.range(0, (batchSize - 1) * beamSize, beamSize):long())) -- batchSize
      for t = targetLength, 1, -1 do
        predTarget[{{1, batchSize}, t}]:copy(currentIndices)
        indices = self.beamParentsHistory[t]:view(-1):index(1, indices + onmt.utils.Cuda.convert(torch.range(0, (batchSize - 1) * beamSize, beamSize):long())) -- batchSize
        if t > 1 then
          currentIndices = self.currentIndicesHistory[t-1]:view(-1):index(1, indices + onmt.utils.Cuda.convert(torch.range(0, (batchSize - 1) * beamSize, beamSize):long())) -- batchSize
        end
      end
      local predLabels = targetsTensorToLabelStrings(predTarget)
      local goldLabels = targetsTensorToLabelStrings(targetOutput)
      local editDistanceRate = evalEditDistanceRate(goldLabels, predLabels)
      numCorrect = batchSize - editDistanceRate
      if self.outputFile then
        for i = 1, #imagePaths do
          self.outputFile:write(string.format('%s\t%s\n', imagePaths[i], predLabels[i]))
        end
        self.outputFile:flush()
      end
      -- get loss
      loss = self.decoder:computeLoss(decoderBatch, nil, context, self.criterion) / batchSize
    else -- isForwardOnly == false
      local _, gradContext, totalLoss = self.decoder:backward(decoderBatch, decoderOutputs, self.criterion)
      loss = totalLoss / batchSize
      gradContext = gradContext:contiguous():view(batchSize, featureMapHeight, featureMapWidth, -1) -- (batchSize, featureMapHeight, featureMapWidth, cnnFeatureSize)
      local gradPadding = onmt.utils.Cuda.convert(torch.zeros(batchSize, featureMapHeight, 1, self.config.cnnFeatureSize))
      gradContext = torch.cat(gradPadding, gradContext, 3)
      gradContext = torch.cat(gradContext, gradPadding, 3)
      local cnnGrad = self.cnnGradProto[{ {1, featureMapHeight}, {1, batchSize}, {1, featureMapWidth}, {} }]
      for i = 1, featureMapHeight do
        local cnnOutput = cnnOutputs[i]
        local source = cnnOutput:transpose(1,2)
        local pos = onmt.utils.Cuda.convert(torch.zeros(batchSize)):fill(i)
        local posEmbeddingFw = self.posEmbeddingFw:forward(pos):view(1, batchSize, -1)
        local posEmbeddingBw = self.posEmbeddingBw:forward(pos):view(1, batchSize, -1)
        source = torch.cat(posEmbeddingFw, source, 1)
        source = torch.cat(source, posEmbeddingBw, 1) -- (featureMapWidth + 2, batchSize, cnnFeatureSize)
        local encoderBatch = Batch():setSourceInput(source)
        local _, rowContext = self.encoder:forward(encoderBatch)
        local rowContextGrad = self.encoder:backward(encoderBatch, nil, gradContext:select(2,i))
        for t = 1, featureMapWidth do
          cnnGrad[{i, {}, t, {}}]:copy(rowContextGrad[t + 1])
        end
        self.posEmbeddingFw:backward(pos, rowContextGrad[1])
        self.posEmbeddingBw:backward(pos, rowContextGrad[featureMapWidth + 2])
      end
      -- cnn
      cnnGrad = cnnGrad:split(1, 1)
      for i = 1, #cnnGrad do
      cnnGrad[i] = cnnGrad[i]:contiguous():view(batchSize, featureMapWidth, -1)
      end
      self.cnn:backward(images, cnnGrad)
      collectgarbage()
    end
    return loss, self.gradParams, {numNonzeros, numCorrect}
  end
  if not isForwardOnly then
    -- optimizer
    local optim = onmt.train.Optim.new(self.optimState)
    optim:zeroGrad(self.gradParams)
    local loss, _, stats = feval(self.params)
    optim:prepareGrad(self.gradParams, 20.0)
    optim:updateParams(self.params, self.gradParams)

    return loss * batchSize, stats
  else
    local loss, _, stats = feval(self.params)
    return loss * batchSize, stats 
  end
end

-- Save model to model_path
function model:save(modelPath)
  for i = 1, #self.layers do
    self.layers[i]:clearState()
  end
  torch.save(modelPath, {{self.cnn, self.encoder:serialize(), self.decoder:serialize(), self.posEmbeddingFw, self.posEmbeddingBw}, self.config, self.numSteps, self.optimState, idToVocab})
end

-- destructor
function model:shutdown()
  if self.outputFile then
    self.outputFile:close()
    logger:info('Results saved to %s.', self.outputPath)
  end
end
