import stanza as st

st.download('arabic') 
pipe = st.Pipeline('arabic')


def get_grams(text, gram):
    text = pipe(text)
    pos = []
    word_pos = []
    returned = []
    
    for i in text.sentences:
        for j in i.words:
            pos.append([j.text, j.pos])

    if gram == 'word':
        returned.append([i[0] for i in pos])
    
    elif gram == 'pos':
        returned.append([i[1] for i in pos])
        

    elif gram == 'word-pos':
        returned.append([i[0]+'-'+i[1] for i in pos])
    
        
    return returned

def _increment(array, index, maximum):
    if array[index] == maximum:
        array[index] = 0
        return _increment(array, index+1, maximum)
    else:
        array[index] += 1
        return array

def _ksngrams(grams, s, n, ksngrams):
    skips = [0]*(n-1)
    while (True):
        ss = sum(skips)
        for gn_i in range(0, len(grams)-(n+ss)+1):
            ksngram = []
            skips_sofar = 0
            ksngram.append(grams[gn_i])
            for ln_i in range(1, n):
                skips_sofar += skips[ln_i-1] + 1
                ksngram.append(grams[gn_i + skips_sofar])
            ksngram_str = '::'.join(item for item in ksngram)
            print()
            if ksngram_str in ksngrams:
                ksngrams[ksngram_str] += 1
            else:
                ksngrams[ksngram_str] = 1
        if ss == s*(n-1):
            break
        else:
           _increment(skips, 0, s)

def getcount_ksngrams(grams, k=0, n=4, minfreq=0, normalize=True, croot=None, tid=None, v=False):
    

    # retrieve freshly
    ksngrams = {}
    for grams_row in grams:
        _ksngrams(grams_row, k, n, ksngrams)

    # filter (normalize, minfreq)
    ksngrams_sum = sum(ksngrams.values())
    ksngrams_filtered = {}
    for key in ksngrams:
        if ksngrams[key] >= minfreq:
            ksngrams_filtered[key] = ksngrams[key]
            if normalize:
                ksngrams_filtered[key] /= float(ksngrams_sum)

    # cache results
    if croot:
        with open(cfile, 'w') as f:
            json.dump(ksngrams_filtered, f)
        if v:
            stderr.write(PREFIX_INFO + 'cache update: file "' + cfile + '" saved.\n')

    return ksngrams_filtered



