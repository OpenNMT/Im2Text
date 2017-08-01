local translate = {}

translate.Advancer = require('opennmt.translate.Advancer')
translate.Beam = require('opennmt.translate.Beam')
translate.BeamSearcher = require('opennmt.translate.BeamSearcher')
translate.DecoderAdvancer = require('opennmt.translate.DecoderAdvancer')
translate.PhraseTable = require('opennmt.translate.PhraseTable')
translate.Translator = require('opennmt.translate.Translator')

return translate
