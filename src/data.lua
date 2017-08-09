 --[[ Load data. Adapted from https://github.com/da03/Attention-OCR/blob/master/src/data_util/data_gen.py.
 --  ARGS:
     * `imageDir`       : string. The base directory of the image path in dataPath.
     * `dataPath`       : string. The file containing data file names and label indexes. Format per line: imagePath[Space]labelIndex. Note that the imagePath is the relative path to imageDir. LabelIndex counts from 0.
     * `labelPath`       : string. The file containing labels. Each line corresponds to a label.
     * `maxImageHeight` : int. Maximum image height. Default: unlimited.
     * `maxImageWidth`  : int. Maximum image width. Default: unlimited.
     * `maxNumTokens`   : int. Maximum number of output tokens. Default: unlimited.
 --]]
require 'image'
require 'paths'
require 'class'
local tds = require('tds')

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(imageDir, dataPath, labelPath, maxImageHeight, maxImageWidth, maxNumTokens)
  self.imageDir = imageDir
  self.labelPath = labelPath
  self.maxImageHeight = maxImageHeight or math.huge
  self.maxImageWidth = maxImageWidth or math.huge
  self.maxNumTokens = maxNumTokens or math.huge

  local file, err = io.open(dataPath, "r")
  if err then
    file, err = io.open(paths.concat(imageDir, dataPath), "r")
    if err then
      _G.logger:error('Data file %s not found', dataPath)
      os.exit()
    end
  end
  self.lines = tds.Hash()
  local idx = 0
  for line in file:lines() do
    idx = idx + 1
    if idx % 1000000 == 0 then
      _G.logger:info ('%d lines read', idx)
    end
    local imagePath, label = unpack(line:split('[%s]+'))
    self.lines[idx] = tds.Vec({imagePath, label})
  end
  self.perm = torch.range(1, #self.lines)
  self.cursor = 1
  self.buffer = {}
  collectgarbage()
end

function DataLoader:shuffle()
  local perm = torch.randperm(#self.lines)
  self.lines = onmt.utils.Table.reorder(self.lines, perm)
  self.perm = self.perm:index(1, perm:type('torch.LongTensor'))
end

function DataLoader:load(dataPath)
  assert(paths.filep(dataPath), string.format('Data file %s does not exist!', dataPath))
  local s
  self.perm, self.cursor, self.buffer, self.epoch, s = table.unpack(torch.load(dataPath))
  self.lines = onmt.utils.Table.reorder(self.lines, self.perm)
  torch.setRNGState(s)
end

function DataLoader:save(dataPath)
  local s = torch.getRNGState()
  torch.save(dataPath, {self.perm, self.cursor, self.buffer, self.epoch, s})
end

function DataLoader:size()
  return #self.lines
end

function DataLoader:nextBatch(batchSize)
  while true do -- accumulate samples of the same size in self.buffer until batchSize or the last data example is reached
    if self.cursor > #self.lines then
      break
    end
    local imagePath = self.lines[self.cursor][1]
    local status, imageData = pcall(image.load, paths.concat(self.imageDir, imagePath))
    if not status then
      self.cursor = self.cursor + 1
      _G.logger:warning('Fails to read image file %s', imagePath)
    else
      local labelIndex = self.lines[self.cursor][2]
      local tokenIds = labelIndexToTokenIds(labelIndex, self.labelPath)
      self.cursor = self.cursor + 1
      imageData = 255.0*image.rgb2y(imageData) -- convert to greyscale
      local imageHeight, imageWidth = imageData:size(2), imageData:size(3)
      if #tokenIds > self.maxNumTokens + 2 then -- truncate target sequence
        _G.logger:warning('Image %s\'s target sequence is too long, will be truncated. Consider using a larger maxNumTokens', imagePath)
        local tokenIdsTemp = {}
        for i = 1, self.maxNumTokens + 2 do
          tokenIdsTemp[i] = tokenIds[i]
        end
        tokenIds = tokenIdsTemp
      end
      if imageHeight <= self.maxImageHeight and imageWidth <= self.maxImageWidth then
        if self.buffer[imageWidth] == nil then
          self.buffer[imageWidth] = {}
        end
        if self.buffer[imageWidth][imageHeight] == nil then
          self.buffer[imageWidth][imageHeight] = {}
        end
        table.insert(self.buffer[imageWidth][imageHeight], {imageData, tokenIds, imagePath})
        if #self.buffer[imageWidth][imageHeight] >= batchSize then
          local images = torch.Tensor(batchSize, 1, imageHeight, imageWidth)
          local maxTargetLength = -math.huge
          local imagePaths = {}
          local offset = #self.buffer[imageWidth][imageHeight] - batchSize
          for i = 1, batchSize do
            imagePaths[i] = self.buffer[imageWidth][imageHeight][i+offset][3]
            images[i]:copy(self.buffer[imageWidth][imageHeight][i+offset][1])
            maxTargetLength = math.max(maxTargetLength, #self.buffer[imageWidth][imageHeight][i+offset][2])
          end
          -- targetInput: used as input to decoder. SOS, tokenId1, tokenId2, ..., tokenIdN
          local targetInput = torch.IntTensor(batchSize, maxTargetLength-1):fill(onmt.Constants.PAD)
          -- targetOutput: used for comparing against decoder's output. tokenId1, tokenId2, ..., tokenIdN, EOS
          local targetOutput = torch.IntTensor(batchSize, maxTargetLength-1):fill(onmt.Constants.PAD)
          local numNonzeros = 0
          for i = 1, batchSize do
            numNonzeros = numNonzeros + #self.buffer[imageWidth][imageHeight][i+offset][2] - 1
            for j = 1, #self.buffer[imageWidth][imageHeight][i+offset][2]-1 do
              targetInput[i][j] = self.buffer[imageWidth][imageHeight][i+offset][2][j]
              targetOutput[i][j] = self.buffer[imageWidth][imageHeight][i+offset][2][j+1]
            end
          end
          if offset == 0 then
            self.buffer[imageWidth][imageHeight] = nil
          else
            for i = 1, batchSize do
              self.buffer[imageWidth][imageHeight][i+offset] = nil
            end
          end

          do return {images, targetInput, targetOutput, numNonzeros, imagePaths} end
        end
      else --  not (imageHeight <= self.maxImageHeight and imageWidth <= self.maxImageWidth)
        _G.logger:warning('Image %s is too large, will be ignored. Consider using a larger maxImageWidth or maxImageHeight'%imagePath)
      end
    end
  end -- cannot accumulate batchSize samples

  -- find if there are any samples left in order to finish the current epoch
  if next(self.buffer) == nil then
    self.cursor = 1
    collectgarbage()
    return nil
  end
  local imageWidth = next(self.buffer)
  while next(self.buffer[imageWidth]) == nil do
    if next(self.buffer, imageWidth) == nil then
      self.cursor = 1
      collectgarbage()
      return nil
    end
    imageWidth = next(self.buffer, imageWidth)
  end
  local imageHeight = next(self.buffer[imageWidth], nil)
  local actualBatchSize = math.min(batchSize, #self.buffer[imageWidth][imageHeight])
  local offset = math.max(0, #self.buffer[imageWidth][imageHeight]-batchSize)
  local images = torch.Tensor(actualBatchSize, 1, imageHeight, imageWidth)
  local maxTargetLength = -math.huge
  local imagePaths = {}
  for i = 1, actualBatchSize do
    imagePaths[i] = self.buffer[imageWidth][imageHeight][i+offset][3]
    images[i]:copy(self.buffer[imageWidth][imageHeight][i+offset][1])
    maxTargetLength = math.max(maxTargetLength, #self.buffer[imageWidth][imageHeight][i+offset][2])
  end
  local targetInput = torch.IntTensor(actualBatchSize, maxTargetLength-1):fill(onmt.Constants.PAD)
  local targetOutput = torch.IntTensor(actualBatchSize, maxTargetLength-1):fill(onmt.Constants.PAD)
  local numNonzeros = 0
  for i = 1, actualBatchSize do
    numNonzeros = numNonzeros + #self.buffer[imageWidth][imageHeight][i+offset][2] - 1
    for j = 1, #self.buffer[imageWidth][imageHeight][i+offset][2]-1 do
      targetInput[i][j] = self.buffer[imageWidth][imageHeight][i+offset][2][j]
      targetOutput[i][j] = self.buffer[imageWidth][imageHeight][i+offset][2][j+1]
    end
  end
  if offset == 0 then
    self.buffer[imageWidth][imageHeight] = nil
  else
    for i = 1, actualBatchSize do
      self.buffer[imageWidth][imageHeight][i+offset] = nil
    end
  end
  return {images, targetInput, targetOutput, numNonzeros, imagePaths}
end

-- convert labelIndex to a list of token ids
function labelIndexToTokenIds(labelIndex, labelPath)
  assert (_G.idToVocab, '_G.idToVocab must be ready before calling labelIndexToTokenIds')
  if _G.vocabToId == nil then
    _G.vocabToId = tds.Hash()
    for i = 1, #_G.idToVocab do
      _G.vocabToId[_G.idToVocab[i]] = i+4
    end
  end
  if labelLines == nil then
    labelLines = tds.Hash()
    local labelFile, err = io.open(labelPath, "r")
    if not err then
      for line in labelFile:lines() do
        local tokenList = (onmt.utils.String.strip(line)):split('[%s]+')
        labelLines[#labelLines+1] = tds.Vec(tokenList)
      end
    end
  end
  local tokenIds = tds.Hash()
  tokenIds[1] = onmt.Constants.BOS
  if labelIndex == nil then
    tokenIds[#tokenIds+1] = onmt.Constants.EOS
    do return tokenIds end
  end
  local tokens = labelLines[tonumber(labelIndex)+1]
  if tokens == nil then
    tokenIds[#tokenIds+1] = onmt.Constants.EOS
    do return tokenIds end
  end
  for i = 1, #tokens do
    local token = tokens[i]
    if _G.vocabToId[token] then
      tokenIds[#tokenIds+1] = _G.vocabToId[token]
    else
      tokenIds[#tokenIds+1] = onmt.Constants.UNK -- unknown token
    end
  end
  tokenIds[#tokenIds+1] = onmt.Constants.EOS
  return tokenIds
end

-- convert targets tensor to a list of label strings
function targetsTensorToLabelStrings(targets)
  assert (targets:dim() == 2)
  local batchSize = targets:size(1)
  local targetLength = targets:size()[2]

  local labelStrings = {}
  for b = 1, batchSize do
    local tokenIds = {}
    for t = 1, targetLength do
      local tokenId = targets[b][t]
      if tokenId == onmt.Constants.EOS then -- ignore tokens after EOS
        break
      end
      table.insert(tokenIds, tokenId)
    end
    local labelString = tokenIdsToLabelString(tokenIds)
    table.insert(labelStrings, labelString)
  end
  return labelStrings
end

-- evaluate the edit distance error rate of the predictions
function evalEditDistanceRate(goldLabelStrings, predLabelStrings)
  assert(#goldLabelStrings == #predLabelStrings)

  local totalEditDistanceRate = 0.0
  for b = 1, #goldLabelStrings do
    local editDistance = string.levenshtein(goldLabelStrings[b], predLabelStrings[b])
    totalEditDistanceRate = totalEditDistanceRate + editDistance / (string.len(goldLabelStrings[b]) + string.len(predLabelStrings[b]))
  end
  return totalEditDistanceRate
end

-- convert a list of token ids to label string
function tokenIdsToLabelString(tokenIds)
  local labelString = tds.Vec()
  for i = 1, #tokenIds do
    local tokenId = tokenIds[i]
    if tokenId == onmt.Constants.PAD or tokenId == onmt.Constants.BOS or tokenId == onmt.Constants.EOS then
      break
    end
    local token = _G.idToVocab[tokenId-4]
    if tokenId == onmt.Constants.UNK then
      token = 'UNK'
    end
    assert (token, 'Make sure your target vocab size is correct!')
    for c in token:gmatch"." do
      labelString:insert(c)
    end
    labelString:insert(' ')
  end
  labelString = labelString:concat()
  return labelString
end


-- https://gist.github.com/Badgerati/3261142
-- Returns the Levenshtein distance between the two given strings
function string.levenshtein(str1, str2)
  local len1 = string.len(str1)
  local len2 = string.len(str2)
  local matrix = {}
  local cost

  -- quick cut-offs to save time
  if (len1 == 0) then
    return len2
  elseif (len2 == 0) then
    return len1
  elseif (str1 == str2) then
    return 0
  end

  -- initialise the base matrix values
  for i = 0, len1, 1 do
    matrix[i] = {}
    matrix[i][0] = i
  end
  for j = 0, len2, 1 do
    matrix[0][j] = j
  end

  -- actual Levenshtein algorithm
  for i = 1, len1, 1 do
    for j = 1, len2, 1 do
      if (str1:byte(i) == str2:byte(j)) then
        cost = 0
      else
        cost = 1
      end
      matrix[i][j] = math.min(matrix[i-1][j] + 1, matrix[i][j-1] + 1, matrix[i-1][j-1] + cost)
    end
  end

  -- return the last value - this is the Levenshtein distance
  return matrix[len1][len2]
end
