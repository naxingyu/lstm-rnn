lstm-rnn
========

Portal of Johannes and Felix's RNN implementation and further modifications for ASR/LVCSR purpose.

This is just a portal!!!! (for now)
The project was open-sourced at SourceForge by Johannes Bergmann and Felix Weninger. https://sourceforge.net/p/currennt

## Changlog
1. Add TIMIT example according to [Alex2013](Alex2013) recipe for phone recognition, differing in that state identities (183) are used instead of phoneme identities (61), requiring frame-level alignment.

2. Modify tools/htk2nc.cpp. For LVCSR tasks, we have a large number of classes and training data. Thus we would want to generate several moderate size NC files instead of a huge one. In such case, there are chances that not all classes appear in one batch of training data. This result in different state mappings for different NC files using the previous htk2nc, which corrupts the network. My modification is simple. Using the physical states indices defined in the label files directly will address this issue. To solve the state adsence problem, the number of physical states must be given. Check the help note in the updated htk2nc code.


[Alex2013] Alex Graves, Navdeep Jaitly and Abdel-rahman Mohamed. Hybrid Speech Recognition with Deep Bidirectional LSTM, ASRU 2013.
