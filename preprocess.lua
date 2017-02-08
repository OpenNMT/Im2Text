require('onmt.init')
require 'image'
local tds = require 'tds'

local cmd = onmt.ExtendedCmdLine.new("preprocess.lua")

local dataType = 'IMG-TEXT'

-------------- Options declaration
local preprocess_options = {
  {'-image_dir', '', [[The base directory of the image path in data-path.]]},
  {'-train', '', [[The file containing data file names and label indexes.
                   Format per line: image_path label_index. Note that label_index
                   counts from 0.]]},
  {'-labels', '', [[The file containing tokenized labels. Each line corresponds
                    to a label.]]},
  {'-valid', '', [[The path containing validate data file names and labels.
                   Format per line: image_path characters]]},
  {'-vocab', '', [[Optional. Vocabulary file. A token per line.]]},
  {'-unk_threshold', 1, [[Optional. If the number of occurences of a token is less
                          than (including) the threshold, then it will be excluded
                          from the generated vocabulary.]]},
  {'-max_image_width', 500, [[Maximum image width]]},
  {'-max_image_height', 160, [[Maximum image height]]},
  {'-max_num_tokens', 150, [[Maximum number of output tokens]]},
  {'-save_data',               '',     [[Output file for the prepared data]]}
}

cmd:setCmdLineOptions(preprocess_options, "Preprocess")

local misc_options = {
  {'-seed',                   910820,    [[Random seed]],
                                   {valid=onmt.ExtendedCmdLine.isUInt()}},
  {'-report_every',           10000,  [[Report status every this many images]],
                                   {valid=onmt.ExtendedCmdLine.isUInt()}}
}
cmd:setCmdLineOptions(misc_options, "Other")
onmt.utils.Logger.declareOpts(cmd)

local opt = cmd:parse(arg)

local function isValid(imageData, tokens, maxImageHeight, maxImageWidth, maxNumTokens)
  local imageHeight, imageWidth = imageData:size(2), imageData:size(3)
  return imageHeight <= maxImageHeight and imageWidth <= maxImageWidth
         and #tokens <= maxNumTokens
end

local function readLabels(labelPath)
  local file, err = io.open(labelPath, "r")
  if err then
    _G.logger:error('Label path %s does not exist!', labelPath)
    os.exit()
  end
  local labels = tds.Vec()
  for line in file:lines() do
    local tokens = tds.Vec((onmt.utils.String.strip(line)):split('[%s]'))
    labels[#labels + 1] = tokens
  end
  return labels
end

local function generateVocab(unk_threshold, dataPath, labels, imageDir,
                             maxImageHeight, maxImageWidth, maxNumTokens)
  local wordFreq = {}
  local dataFile, dataErr = io.open(dataPath, "r")
  if dataErr then
    _G.logger:error('Data path %s does not exist!', dataPath)
    os.exit()
  end
  for line in dataFile:lines() do
    local imagePath, labelId = unpack(line:split('[%s]+'))
    local tokens = labels[tonumber(labelId) + 1]
    local status, imageData = pcall(image.load, paths.concat(imageDir, imagePath))
    if status then
      if isValid(imageData, tokens, maxImageHeight, maxImageWidth, maxNumTokens) then
        for _, token in ipairs(tokens) do
          if token:len() == 0 then
            token = ' '
          end
          wordFreq[token] = wordFreq[token] or 0
          wordFreq[token] = wordFreq[token] + 1
        end
      end
    else
      _G.logger:warning('Fails to read image file %s', imagePath)
    end
  end

  local vocab = onmt.utils.Dict.new({onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                                     onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD})
  for token, freq in pairs(wordFreq) do
    if freq > unk_threshold then
      vocab:add(token, vocab:size() + 1)
    end
  end
  return vocab
end

local function preprocessData(dataPath, imageDir, labels, vocab,
                              maxImageHeight, maxImageWidth, maxNumTokens)
  local file, err = io.open(dataPath, "r")
  if err then
    file, err = io.open(paths.concat(imageDir, dataPath), "r")
    if err then
      _G.logger:error('Data file %s not found', dataPath)
      os.exit()
    end
  end

  local src = tds.Vec()
  local tgt = tds.Vec()
  local sizes = {}
  local idx = 0
  local pruned = 0
  local invalid = 0
  for line in file:lines() do
    idx = idx + 1
    if idx % 1000000 == 0 then
      _G.logger:info ('%d lines read', idx)
    end
    local imagePath, labelId = unpack(onmt.utils.String.strip(line):split('[%s]+'))
    local status, imageData = pcall(image.load, paths.concat(imageDir, imagePath))
    if not status then
      _G.logger:warning('Fails to read image file %s.', imagePath)
      invalid = invalid + 1
    else
      -- Convert to grayscale
      imageData = 255.0 * image.rgb2y(imageData)
      local label = labels[tonumber(labelId) + 1]
      if isValid(imageData, label, maxImageHeight, maxImageWidth, maxNumTokens) then
        local tokens = vocab:convertToIdx(label, onmt.Constants.UNK_WORD,
                               onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD)
        local h = imageData:size(2)
        local w = imageData:size(3)
        src[#src + 1] = imageData
        tgt[#tgt + 1] = tokens
        sizes[#sizes + 1] = tds.Vec({#sizes + 1, h, w})
      else
        _G.logger:warning('Image file %s pruned due to size constraints.', imagePath)
        pruned = pruned + 1
      end
    end
  end

  -- Sort images w.r.t size
  _G.logger:info('Sorting')
  table.sort(sizes, function(a,b) return (a[2]>b[2]) or (a[2]==b[2] and a[3]>b[3]) end)
  local srcNew = tds.Hash()
  local tgtNew = tds.Hash()
  for i = 1, #sizes do
    srcNew[i] = src[sizes[i][1]]
    tgtNew[i] = tgt[sizes[i][1]]
  end
  _G.logger:info('%d processed, %d invalid images, %d pruned', idx, invalid, pruned)
  return srcNew, tgtNew
end

local function main()

  torch.manualSeed(opt.seed)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local data = {}
  data.dataType = dataType

  -- Read labels
  local labels = readLabels(opt.labels)

  local vocab
  if opt.vocab:len() == 0 then
    -- Generate vocabulary
    _G.logger:info('Generating vocabulary from training data...')
    vocab = generateVocab(opt.unk_threshold, opt.train, labels, opt.image_dir,
                          opt.max_image_height, opt.max_image_width, opt.max_num_tokens)
    -- Save vocabulary
    _G.logger:info('Saving vocabulary to %s', opt.save_data .. '.dict')
    vocab:writeFile(opt.save_data .. '.dict')
  else
    -- Load vocabulary
    _G.logger:info('Loading vocabulary from %s', opt.vocab)
    vocab = onmt.utils.Dict.new(opt.vocab)
  end
  data.dicts = vocab

  _G.logger:info('')

  _G.logger:info('Preparing training data...')
  data.train = {}
  data.train.src, data.train.tgt = {}, {}
  data.train.src.features, data.train.tgt.features = {}, {}
  data.train.src.words, data.train.tgt.words = preprocessData(opt.train, opt.image_dir, labels, vocab,
                             opt.max_image_height, opt.max_image_width, opt.max_num_tokens)

  _G.logger:info('')

  _G.logger:info('Preparing validation data...')
  data.valid = {}
  data.valid.src, data.valid.tgt = {}, {}
  data.valid.src.features, data.valid.tgt.features = {}, {}
  data.valid.src.words, data.valid.tgt.words = preprocessData(opt.valid, opt.image_dir, labels, vocab,
                             opt.max_image_height, opt.max_image_width, opt.max_num_tokens)

  _G.logger:info('')

  _G.logger:info('Saving data to \'' .. opt.save_data .. '-train.t7\'...')
  torch.save(opt.save_data .. '-train.t7', data, 'binary', false)
  _G.logger:shutDown()
end

main()
