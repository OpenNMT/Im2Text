onmt = onmt or {}

require('opennmt.modules.Sequencer')
require('opennmt.modules.Encoder')
require('opennmt.modules.BiEncoder')
require('opennmt.modules.DBiEncoder')
require('opennmt.modules.PDBiEncoder')
require('opennmt.modules.Decoder')

require('opennmt.modules.Network')

require('opennmt.modules.GRU')
require('opennmt.modules.LSTM')

require('opennmt.modules.MaskedSoftmax')
require('opennmt.modules.WordEmbedding')
require('opennmt.modules.FeaturesEmbedding')

require('opennmt.modules.NoAttention')
require('opennmt.modules.GlobalAttention')

require('opennmt.modules.Generator')
require('opennmt.modules.FeaturesGenerator')

require('opennmt.modules.JoinReplicateTable')
require('opennmt.modules.ParallelClassNLLCriterion')

return onmt
