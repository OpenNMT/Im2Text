 --[[ Training, adapted from https://github.com/harvardnlp/seq2seq-attn/blob/master/train.lua
--]]
require 'nn'
require 'nngraph'
require 'paths'
local status = pcall(function() require('onmt.init') end)
if not status then
  print('OpenNMT not found. Please enter the path to OpenNMT: ')
  local onmtPath = io.read()
  package.path = package.path .. ';' .. paths.concat(onmtPath, '?.lua')
  status = pcall(function() require('onmt.init') end)
  if not status then
    print ('Error: onmt not found in the specified path!')
    os.exit(1)
  end
end
require 'onmt.models.Model'
local tds = require 'tds'

local cmd = onmt.ExtendedCmdLine.new("src/train.lua")

-------------- Options declaration
local data_options = {
  {'-data',       '', [[Path to the training *-train.t7 file from preprocess.lua]],
                      {valid=onmt.ExtendedCmdLine.nonEmpty}},
  {'-save_model', '', [[Model filename (the model will be saved as
                            <save_model>_epochN_PPL.t7 where PPL is the validation perplexity]],
                      {valid=onmt.ExtendedCmdLine.nonEmpty}}
}

cmd:setCmdLineOptions(data_options, "Data")


-- Generic Model options.
onmt.Model.declareOpts(cmd)

-- Optimization options.
onmt.train.Optim.declareOpts(cmd)

-- Training process options.
onmt.Trainer.declareOpts(cmd)

-- Checkpoints options.
onmt.train.Checkpoint.declareOpts(cmd)

-- GPU
onmt.utils.Cuda.declareOpts(cmd)
-- Memory optimization
onmt.utils.Memory.declareOpts(cmd)
-- Misc
cmd:option('-seed', 3435, [[Seed for random initialization]], {valid=onmt.ExtendedCmdLine.isUInt()})
-- Logger options
onmt.utils.Logger.declareOpts(cmd)
-- Profiler options
onmt.utils.Profiler.declareOpts(cmd)
-- Optimization options.
onmt.train.Optim.declareOpts(cmd)

local modelClass = require 'src.model'
modelClass.declareOpts(cmd)

local opt = cmd:parse(arg)

local function main()
  torch.manualSeed(opt.seed)
  math.randomseed(opt.seed)

  _G.logger = onmt.utils.Logger.new(opt.log_path)
  _G.profiler = onmt.utils.Profiler.new(opt.profile)
  _G.logger:info('Command Line Arguments: %s', table.concat(arg, ' ') or '')

  onmt.utils.Cuda.init(opt)
  onmt.utils.Parallel.init(opt)


  local checkpoint
  if onmt.utils.Cuda.activated then
    require 'cudnn'
  end
  checkpoint, opt = onmt.train.Checkpoint.loadFromCheckpoint(opt)

  local dataset = torch.load(opt.data, 'binary', false)

  -- main model
  local model

  -- build or load model from checkpoint and copy to GPUs
  onmt.utils.Parallel.launch(function(idx)
    if checkpoint.models then
      _G.model = modelClass.new(opt, checkpoint, idx > 1)
    else
      local verbose = idx == 1
      _G.model = modelClass.new(opt, dataset, verbose)
    end
    onmt.utils.Cuda.convert(_G.model)
    return idx, _G.model
  end, function(idx, themodel)
    if idx == 1 then
      model = themodel
    end
  end)
  -- Define optimization method.
  local optim = onmt.train.Optim.new(opt, opt.optim_states)
  -- Initialize trainer.
  local trainer = onmt.Trainer.new(opt)


  -- keep backward compatibility
  dataset.dataType = dataset.dataType or "BITEXT"

  local Batch = onmt.data.Batch
  
  --[[ Return the maxLength, sizes, and non-zero count
    of a batch of `seq`s ignoring `ignore` words.
  --]]
  local function getLength(seq, ignore)
    if #seq == 0 then
      return 0, 0, 0
    end
    local ndim = #seq[1]:size():totable()
    local sizes = torch.IntTensor(#seq, ndim):zero()
    local max = torch.Tensor(ndim):zero()
    local sum = 0
  
    for i = 1, #seq do
      local len = torch.Tensor(seq[i]:size():totable())
      if ignore ~= nil then
        len = len:add(-ignore)
      end
      max = torch.cmax(max, len)
      sum = sum + len:prod()
      sizes[i]:copy(len)
    end
    if ndim == 1 then
      max = max[1]
      sizes = sizes:view(-1)
    end
    return max, sizes, sum
  end
  --[[ Create a batch object.
  
  Parameters:
  
    * `src` - 2D table of source batch indices
    * `srcFeatures` - 2D table of source batch features (opt)
    * `tgt` - 2D table of target batch indices
    * `tgtFeatures` - 2D table of target batch features (opt)
  --]]
  function Batch:__init(src, srcFeatures, tgt, tgtFeatures)
    src = src or {}
    srcFeatures = srcFeatures or {}
    tgtFeatures = tgtFeatures or {}
  
    if tgt ~= nil then
      assert(#src == #tgt, "source and target must have the same batch size")
    end
  
    self.size = #src
  
    self.sourceLength, self.sourceSize = getLength(src)
  
    local sourceSeq
    if torch.isTensor(self.sourceLength) then
      local sizes = self.sourceLength:view(-1):totable()
      sizes[#sizes + 1] = self.size
      sourceSeq = torch.IntTensor(table.unpack(sizes)):fill(onmt.Constants.PAD)
    else
      sourceSeq = torch.IntTensor(self.sourceLength, self.size):fill(onmt.Constants.PAD)
    end
    self.sourceInput = sourceSeq:clone()
    self.sourceInputRev = sourceSeq:clone()
  
    self.sourceInputFeatures = {}
    self.sourceInputRevFeatures = {}
  
    if #srcFeatures > 0 then
      for _ = 1, #srcFeatures[1] do
        table.insert(self.sourceInputFeatures, sourceSeq:clone())
        table.insert(self.sourceInputRevFeatures, sourceSeq:clone())
      end
    end
  
    if tgt ~= nil then
      self.targetLength, self.targetSize, self.targetNonZeros = getLength(tgt, 1)
  
      local targetSeq = torch.IntTensor(self.targetLength, self.size):fill(onmt.Constants.PAD)
      self.targetInput = targetSeq:clone()
      self.targetOutput = targetSeq:clone()
  
      self.targetInputFeatures = {}
      self.targetOutputFeatures = {}
  
      if #tgtFeatures > 0 then
        for _ = 1, #tgtFeatures[1] do
          table.insert(self.targetInputFeatures, targetSeq:clone())
          table.insert(self.targetOutputFeatures, targetSeq:clone())
        end
      end
    end
  
    for b = 1, self.size do
      if torch.isTensor(self.sourceLength) then
        self.sourceInput:narrow(self.sourceLength:size(1)+1, b, 1):copy(src[b])
        self.sourceInputRev:narrow(self.sourceLength:size(1)+1, b, 1):copy(src[b])
      else
        local sourceOffset = self.sourceLength - self.sourceSize[b] + 1
        local sourceInput = src[b]
        local sourceInputRev = src[b]:index(1, torch.linspace(self.sourceSize[b], 1, self.sourceSize[b]):long())
  
        -- Source input is left padded [PPPPPPABCDE] .
        self.sourceInput[{{sourceOffset, self.sourceLength}, b}]:copy(sourceInput)
        self.sourceInputPadLeft = true
  
        -- Rev source input is right padded [EDCBAPPPPPP] .
        self.sourceInputRev[{{1, self.sourceSize[b]}, b}]:copy(sourceInputRev)
        self.sourceInputRevPadLeft = false
  
        for i = 1, #self.sourceInputFeatures do
          local sourceInputFeatures = srcFeatures[b][i]
          local sourceInputRevFeatures = srcFeatures[b][i]:index(1, torch.linspace(self.sourceSize[b], 1, self.sourceSize[b]):long())
  
          self.sourceInputFeatures[i][{{sourceOffset, self.sourceLength}, b}]:copy(sourceInputFeatures)
          self.sourceInputRevFeatures[i][{{1, self.sourceSize[b]}, b}]:copy(sourceInputRevFeatures)
        end
      end
  
      if tgt ~= nil then
        -- Input: [<s>ABCDE]
        -- Ouput: [ABCDE</s>]
        local targetLength = tgt[b]:size(1) - 1
        local targetInput = tgt[b]:narrow(1, 1, targetLength)
        local targetOutput = tgt[b]:narrow(1, 2, targetLength)
  
        -- Target is right padded [<S>ABCDEPPPPPP] .
        self.targetInput[{{1, targetLength}, b}]:copy(targetInput)
        self.targetOutput[{{1, targetLength}, b}]:copy(targetOutput)
  
        for i = 1, #self.targetInputFeatures do
          local targetInputFeatures = tgtFeatures[b][i]:narrow(1, 1, targetLength)
          local targetOutputFeatures = tgtFeatures[b][i]:narrow(1, 2, targetLength)
  
          self.targetInputFeatures[i][{{1, targetLength}, b}]:copy(targetInputFeatures)
          self.targetOutputFeatures[i][{{1, targetLength}, b}]:copy(targetOutputFeatures)
        end
      end
    end
  end
  local Dataset = onmt.data.Dataset
  --[[ Setup up the training data to respect `maxBatchSize`. ]]
  function Dataset:setBatchSize(maxBatchSize)
  
    self.batchRange = {}
    self.maxSourceSize = torch.Tensor(self.src[1]:size():totable())
    self.maxTargetLength = 0
  
    -- Prepares batches in terms of range within self.src and self.tgt.
    local offset = 0
    local batchSize = 1
    local sourceSize = nil
    local targetLength = nil
    for i = 1, #self.src do
      -- Set up the offsets to make same source size batches of the
      -- correct size.
      if batchSize == maxBatchSize or i == 1 or
          torch.Tensor(self.src[i]:size():totable()):ne(sourceSize):any() then
        if i > 1 then
          table.insert(self.batchRange, { ["begin"] = offset, ["end"] = i - 1 })
        end
  
        offset = i
        batchSize = 1
        sourceSize = torch.Tensor(self.src[i]:size():totable())
        targetLength = 0
      else
        batchSize = batchSize + 1
      end
  
      self.maxSourceSize = torch.cmax(self.maxSourceSize, sourceSize)
  
      if self.tgt ~= nil then
        -- Target contains <s> and </s>.
        local targetSeqLength = self.tgt[i]:size(1) - 1
        targetLength = math.max(targetLength, targetSeqLength)
        self.maxTargetLength = math.max(self.maxTargetLength, targetSeqLength)
      end
    end
    -- Catch last batch.
    table.insert(self.batchRange, { ["begin"] = offset, ["end"] = #self.src })
  end
  local trainData = Dataset.new(dataset.train.src, dataset.train.tgt)
  local validData = Dataset.new(dataset.valid.src, dataset.valid.tgt)
  trainData:setBatchSize(opt.max_batch_size)
  validData:setBatchSize(opt.max_batch_size)
  -- Launch train
  trainer:train(model, optim, trainData, validData, dataset, checkpoint.info)

  _G.logger:shutDown()
end -- function main

main()
