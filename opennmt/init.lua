require('torch')

onmt = {}

onmt.utils = require('opennmt.utils.init')

require('opennmt.modules.init')

onmt.data = require('opennmt.data.init')
onmt.train = require('opennmt.train.init')
onmt.translate = require('opennmt.translate.init')
onmt.tagger = require('opennmt.tagger.init')

onmt.Constants = require('opennmt.Constants')
onmt.Factory = require('opennmt.Factory')
onmt.Model = require('opennmt.Model')
onmt.Seq2Seq = require('opennmt.Seq2Seq')
onmt.LanguageModel = require('opennmt.LanguageModel')
onmt.SeqTagger = require('opennmt.SeqTagger')
onmt.ModelSelector = require('opennmt.ModelSelector')

return onmt
