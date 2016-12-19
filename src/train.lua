 --[[ Training, adapted from https://github.com/harvardnlp/seq2seq-attn/blob/master/train.lua
--]]
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'paths'
local status, err = pcall(function() require('onmt') end)
if not status then
  print('OpenNMT not found. Please enter the path to OpenNMT: ')
  local onmtPath = io.read()
  package.path = package.path .. ';' .. paths.concat(onmtPath, '?.lua')
  status, err = pcall(function() require('onmt.init') end)
  if not status then
    print ('Error: onmt not found in the specified path!')
    os.exit(1)
  end
end
tds = require 'tds'

require 'src.model'
require 'src.data'
require 'src.log'

cmd = torch.CmdLine()

-- Input and Output
cmd:text('')
cmd:text('**Control**')
cmd:text('')
cmd:option('-phase', 'test', [[train or test]])
cmd:option('-load_model', false, [[Load model from model_dir or not]])
cmd:option('-gpu_id', 1, [[Which gpu to use]])

cmd:text('')
cmd:text('**Input and Output**')
cmd:text('')
cmd:option('-image_dir', '', [[The base directory of the image path in data-path.]])
cmd:option('-data_path', '', [[The file containing data file names and label indexes. Format per line: image_path label_index. Note that label_index count from 0.]])
cmd:option('-label_path', '', [[The file containing tokenized labels. Each line corresponds to a label.]])
cmd:option('-val_data_path', '', [[The path containing validate data file names and labels. Format per line: image_path characters]])
cmd:option('-vocab_file', '', [[Vocabulary file. A token per line.]])
cmd:option('-model_dir', 'model', [[The directory for saving and loading model parameters (structure is not stored)]])
cmd:option('-output_dir', 'results', [[The path to put results]])

-- Logging
cmd:text('')
cmd:text('**Display**')
cmd:text('')
cmd:option('-steps_per_checkpoint', 100, [[Checkpointing (print perplexity, save model) per how many steps]])
cmd:option('-log_path', 'log.txt', [[The path to put log]])

-- Optimization
cmd:text('')
cmd:text('**Optimization**')
cmd:text('')
cmd:option('-num_epochs', 15, [[The number of whole data passes]])
cmd:option('-batch_size', 1, [[Batch size]])
cmd:option('-learning_rate', 0.1, [[Initial learning rate]])
cmd:option('-lr_decay', 0.5, [[Decay learning rate by this much if perplexity does not decrease on the validation set]])

-- Network
cmd:text('')
cmd:text('**Network**')
cmd:text('')
cmd:option('-input_feed', false, [[Whether or not use LSTM attention decoder cell]])
cmd:option('-encoder_num_hidden', 256, [[Number of hidden units in encoder cell]])
cmd:option('-encoder_num_layers', 1, [[Number of hidden layers in encoder cell]])
cmd:option('-decoder_num_layers', 1, [[Number of hidden units in decoder cell]])
cmd:option('-target_embedding_size', 80, [[Embedding dimension for each target]])

-- Other
cmd:text('')
cmd:text('**Other**')
cmd:text('')
cmd:option('-beam_size', 1, [[Beam size for decoding]])
cmd:option('-max_num_tokens', 150, [[Maximum number of output tokens]]) -- when evaluate, this is the cut-off length.
cmd:option('-max_image_width', 500, [[Maximum image width]]) --800/2/2/2
cmd:option('-max_image_height', 160, [[Maximum image height]]) --80 / (2*2*2)
cmd:option('-seed', 910820, [[Load model from model_dir or not]])

local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
math.randomseed(opt.seed)
cutorch.manualSeed(opt.seed)

function run(model, phase, batchSize, numEpochs, trainData, valData, modelDir, stepsPerCheckpoint, beamSize, outputDir, learningRateInit, learningRateDecay)
  local loss = 0
  local numSamples = 0
  local numNonzeros = 0
  model.optimState.learningRate = model.optimState.learningRate or learningRateInit
  logger:info('Learning Rate: %f', model.optimState.learningRate)

  assert(phase == 'train' or phase == 'test', 'phase must be either train or test')
  local isForwardOnly
  if phase == 'train' then
    isForwardOnly = false
  else
    isForwardOnly = true
    numEpochs = 1
    model.numSteps = 0
    model:setOutputDirectory(outputDir)
  end

  logger:info('Running...')
  local valLosses = {}
  -- Run numEpochs epochs
  for epoch = 1, numEpochs do
    if not isForwardOnly then
      trainData:shuffle()
    end
    -- Run 1 epoch
    while true do
      trainBatch = trainData:nextBatch(batchSize)
      if trainBatch == nil then
        break
      end
      local actualBatchSize = trainBatch[1]:size(1)
      local stepLoss, stats = model:step(trainBatch, isForwardOnly, beamSize) -- do one step
      if not isForwardOnly then
        logger:info('step perplexity: %f', math.exp(stepLoss/stats[1]))
      end
      numSamples = numSamples + actualBatchSize
      numNonzeros = numNonzeros + stats[1]
      loss = loss + stepLoss
      model.numSteps = model.numSteps + 1
      if model.numSteps % stepsPerCheckpoint == 0 then
        if isForwardOnly then
          logger:info('Step: %d. Number of samples: %d.', model.numSteps, numSamples)
        else
          logger:info('Step: %d. Training Perplexity: %f', model.numSteps, math.exp(loss/numNonzeros))
          logger:info('Saving Model')
          local modelPath = paths.concat(modelDir, string.format('model_%d', model.numSteps))
          local modelPathTemp = paths.concat(modelDir, '.model.tmp') -- to ensure atomic operation
          local modelPathLatest = paths.concat(modelDir, 'model_latest')
          model:save(modelPath)
          logger:info('Model saved to %s', modelPath)
          os.execute(string.format('cp %s %s', modelPath, modelPathTemp))
          os.execute(string.format('mv %s %s', modelPathTemp, modelPathLatest))

          loss, numNonzeros = 0, 0, 0
          collectgarbage()
        end
      end
    end -- Run 1 epoch
    -- After each epoch, evaluate on validation if phase is train
    if not isForwardOnly then
      logger:info('Evaluating model on validation data')
      local valLoss, valNumSamples, valNumNonzeros, valNumCorrect = 0, 0, 0, 0
      -- Run 1 epoch on validation data
      while true do
        valBatch = valData:nextBatch(batchSize)
        if valBatch == nil then
          break
        end
        local actualBatchSize = valBatch[1]:size(1)
        local stepLoss, stats = model:step(valBatch, true, beamSize)
        valLoss = valLoss + stepLoss
        valNumSamples = valNumSamples + actualBatchSize
        valNumNonzeros = valNumNonzeros + stats[1]
        valNumCorrect = valNumCorrect + stats[2]
      end -- Run 1 epoch
      valLosses[epoch] = valLoss
      logger:info('Epoch: %d. Step: %d. Val Accuracy: %f. Val Perplexity: %f', epoch, model.numSteps, valNumCorrect/valNumSamples, math.exp(valLoss/valNumNonzeros))
      -- Decay learning rate if validation loss does not decrease
      if valLosses[epoch-1] and valLosses[epoch] > valLosses[epoch-1] then
        model.optimState.learningRate = model.optimState.learningRate * learningRateDecay
        logger:info('Decay learning rate to %f', model.optimState.learningRate)
      end
      logger:info('Saving Model')
      local modelPath = paths.concat(modelDir, string.format('model_%d', model.numSteps))
      local modelPathTemp = paths.concat(modelDir, '.model.tmp')
      local modelPathLatest = paths.concat(modelDir, 'model_latest')
      model:save(modelPath)
      logger:info('Model saved to %s', modelPath)
      os.execute(string.format('cp %s %s', modelPath, modelPathTemp))
      os.execute(string.format('mv %s %s', modelPathTemp, modelPathLatest))
    else -- isForwardOnly == true
      logger:info('Epoch ends. Number of samples: %d.', numSamples)
    end
  end -- for epoch
end -- run function

function main()
  assert (opt.gpu_id > 0, 'Only support using GPU! Please specify a valid gpu_id.')

  logger = logging(opt.log_path)
  logger:info('Command Line Arguments: %s', table.concat(arg, ' ') or '')

  local gpuId = opt.gpu_id
  logger:info('Using CUDA on GPU %d', gpuId)
  cutorch.setDevice(gpuId)
  onmt.utils.Cuda.init({gpuid=gpuId})

  -- Convert Options
  opt.maxDecoderLength = opt.max_num_tokens + 1 -- since <StartOfSequence> is prepended to the sequence
  opt.maxEncoderLengthWidth = math.floor(opt.max_image_width / 8.0) -- feature maps after CNN become 8 times smaller
  opt.maxEncoderLengthHeight = math.floor(opt.max_image_height / 8.0) -- feature maps after CNN become 8 times smaller

  -- Build Model
  logger:info('Building model')
  local model = Model()
  local modelPath = paths.concat(opt.model_dir, 'model_latest')
  if opt.load_model and paths.filep(modelPath) then
    logger:info('Loading model from %s', modelPath)
    model:load(modelPath, opt)
  else
    -- Load Vocabulary
    logger:info('Loading vocabulary from %s', opt.vocab_file)
    idToVocab = tds.Hash() -- vocabulary file is global
    local file, err = io.open(opt.vocab_file, "r")
    if err then
      logger:error('Vocabulary file %s does not exist!', opt.vocab_file)
      os.exit()
    end
    for line in file:lines() do
      local token = onmt.utils.String.strip(line)
      if onmt.utils.String.isEmpty(token) then
        token = ' '
      end
      idToVocab[#idToVocab+1] = token
    end
    opt.targetVocabSize = #idToVocab + 4
    logger:info('Creating model with fresh parameters')
    model:create(opt)
  end

  if not paths.dirp(opt.model_dir) then
    paths.mkdir(opt.model_dir)
  end

  -- Load Data
  logger:info('Image directory: %s', opt.image_dir)
  logger:info('Loading %s data from %s', opt.phase, opt.data_path)
  if opt.phase == 'train' and (not paths.filep(opt.label_path)) then
      logger:error('Label file %s does not exist!', opt.label_path)
      os.exit(1)
  end
  local trainData = DataLoader(opt.image_dir, opt.data_path, opt.label_path, opt.max_image_height, opt.max_image_width, opt.max_num_tokens)
  logger:info('Loaded')
  local valData
  if opt.phase == 'train' then
    logger:info('Loading validation data from %s', opt.val_data_path)
    valData = DataLoader(opt.image_dir, opt.val_data_path, opt.label_path, opt.max_image_height, opt.max_image_width, opt.max_num_tokens)
    logger:info('Loaded')
  end

  -- Run Model
  run(model, opt.phase, opt.batch_size, opt.num_epochs, trainData, valData, opt.model_dir, opt.steps_per_checkpoint, opt.beam_size, opt.output_dir, opt.learning_rate, opt.lr_decay)

  model:shutdown()
  logger:shutdown()
end -- function main

main()
