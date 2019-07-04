import os, sys
from subprocess import call
import numpy as np
from itertools import groupby

import tensorflow as tf
import re

from struct import unpack
import pickle


plotting = False

# Simplest parallelisation:
num_parallel = 1
par_num = 0

if len(sys.argv) > 2:
    num_parallel = int(sys.argv[1])
    par_num = int(sys.argv[2])

print("Parallel thread %i of %i" % (par_num, num_parallel))


'''

We'll want to create tfrecord files containing an approximately identical number of items, and 
for every item we'll want the following features/values:

1 x Audio waveform data (float)
1 x OpenSmile voiceProb_sma; (float)
1 x OpenSmile F0_sma; (float)
1 x OpenSmile pcm_intensity_sma; (float)
1 x OpenSmile pcm_loudness_sma (float)
1 x OpenSmile frame times (float)

1 x Language (String)
1 x Phoneme sequence (int)
1 x phoneme start times (float)

1 x PDF sequence (int)
1 x PDF start times (float)

1 x Letter sequence (String)
1 x word sequence  (String)

1 x Speaker (String)
1 x Sourcefile (String)
'''

debug = False

split_files = False
approx_num_items_in_record = 1000

kaldidir = 'multicorpora_all_corpora' #'multicorpora_all_corpora_duration_deaf'

data_cache_dir = '/teamwork/t40511_asr/scratch/rkarhila/corpora_for_new_siak/new10_tfrecords/'

phonesetfile='/l/rkarhila/kaldi-trunk/egs/'+kaldidir+'/s5/data/lang/phones.txt'

labelbasedir='/l/rkarhila/kaldi-trunk/egs/'+kaldidir+'/s5/data'
modelfile='/l/rkarhila/kaldi-trunk/egs/'+kaldidir+'/s5/exp/tri3b/final.mdl'

promptsfile = '/teamwork/t40511_asr/scratch/rkarhila/corpora_for_new_siak/kaldi_necessities/all_corpora_all_files/data/text'
filescpfile = '/teamwork/t40511_asr/scratch/rkarhila/corpora_for_new_siak/kaldi_necessities/all_corpora_all_files/data/wav.scp'

ipamapfile='/l/rkarhila/another_rnn_phone_classifier/source_data/phonesets/phone_sets.out.pickle'

alignmentbasedir='/l/rkarhila/kaldi-trunk/egs/'+kaldidir+'/s5/exp/'

#modelfile='/l/rkarhila/kaldi-trunk/egs/'+kaldidir+'/s5/exp/tri3b/final.mdl'

#alignmentdir='/l/rkarhila/kaldi-trunk/egs/'+kaldidir+'/s5/exp/tri3b_ali_sp_cleaned_train/'

duration_cutoff_kaldidir = 'multicorpora_finnish_only'
duration_cutoff_pickle = duration_cutoff_kaldidir+'phone_cutoffs.pickle'


ipadefinitions = pickle.load(open(ipamapfile, 'rb'))

duration_cutoffs = pickle.load(open(duration_cutoff_pickle, 'rb'))['cutoffs']

lengthening_mark = 'ː'

alignmentdirs= {'train' : ['tri3b_ali_train_pfstar',
                           'tri3b_ali_train_speechdat-fi',
                           'tri3b_ali_train_speecon-fi',
                           'tri3b_ali_train_speeconkids-fi',
                           'tri3b_ali_train_spraakbanken-se',
                           'tri3b_ali_train_tidigits',
                           'tri3b_ali_train_wsj1',
                           'tri3b_ali_train_wsjcam0' ],
                'devel' : ["tri3b_ali_devel_wsjcam0_si_dt_1_clean_wv1",
                           "tri3b_ali_devel_pfstar_enuk",
                           "tri3b_ali_devel_speechdat-fi_tele_FIA",
                           "tri3b_ali_devel_speecon-fi_cleanFI0",
                           "tri3b_ali_devel_speecon-fi_cleanFI1",
                           "tri3b_ali_devel_speecon-fi_noisy_carFI0",
                           "tri3b_ali_devel_speecon-fi_noisy_carFI1",
                           "tri3b_ali_devel_speecon-fi_noisy_publicFI0",
                           "tri3b_ali_devel_speecon-fi_noisy_publicFI1",
                           "tri3b_ali_devel_speeconkids-fi_cleanF0",
                           "tri3b_ali_devel_speeconkids-fi_cleanF1",
                           "tri3b_ali_devel_spraakbanken-se",
                           "tri3b_ali_devel_wsj1_si_dt_05_wv1",
                           "tri3b_ali_devel_wsj1_si_dt_05_wv2",
                           "tri3b_ali_devel_wsjcam0_si_dt_1_clean_wv2",
                           "tri3b_ali_test_tidigits_children" ],
                'test' : [ "tri3b_ali_test_tidigits_children",               
                           "tri3b_ali_test_pfstar_enuk",
                           "tri3b_ali_test_speechdat-fi",
                           "tri3b_ali_test_speecon-fi_cleanFI0",
                           "tri3b_ali_test_speecon-fi_cleanFI1",
                           "tri3b_ali_test_speecon-fi_noisy_carF0",
                           "tri3b_ali_test_speecon-fi_noisy_carF1",
                           "tri3b_ali_test_speecon-fi_noisy_publicF0",
                           "tri3b_ali_test_speecon-fi_noisy_publicF1",
                           "tri3b_ali_test_speeconkids-fi_cleanF0",
                           "tri3b_ali_test_speeconkids-fi_cleanF1",
                           "tri3b_ali_test_spraakbanken-se",
                           "tri3b_ali_test_wsj1_si_et_h2_wv2",
                           "tri3b_ali_test_wsjcam0_si_et_1_clean_wv1",
                           "tri3b_ali_test_wsjcam0_si_et_1_clean_wv2",
                           "tri3b_ali_test_tidigits_children" ],
                'siak' : [ "tri3b_ali_devel_siak_fi_en_uk",
                           "tri3b_ali_test_siak_fi_en_uk" ],
                'tellme' : [ "tri3b_ali_devel_tellme",
                             "tri3b_ali_test_tellme" ]}


ali_to_phones='/l/rkarhila/kaldi-trunk/src/bin/ali-to-phones'

ali_to_pdfs='/l/rkarhila/kaldi-trunk/src/bin/ali-to-pdf'


#./SMILExtract -C prosodyAcf.conf -I input.wav -O output.csv
opensmile_extractor = '/l/rkarhila/SMILExtract/SMILExtract'
opensmile_config = '/l/rkarhila/SMILExtract/prosodyAcf.conf'
#opensmile_features=np.genfromtxt('/tmp/output.csv', delimiter=';', skip_header=1)
# Opensmile features:
# 0 frameIndex;
# 1 frameTime;
# 2 voiceProb_sma;
# 3 F0_sma;
# 4 pcm_intensity_sma;
# 5 pcm_loudness_sma


#ali_to_phones='ali-to-phones'

timebins = [ [0.1, 2.0],
             [2.0, 4.0],
             [4.0, 6.0],
             [6.0, 8.0],
             [8.0, 10.0],
             [10.0, 12.0],
             [12.0, 18.0],
             [18.0, 30.0] ]

phonecountbins = [ [1,6],
                   [6,12],
                   [12,20],
                   [20,40],
                   [40,80],
                   [80,200],
                   [200,10000] ]

padding_s = 0.04
fs = 16000
bytes_per_sample = 2

utts_per_record = 3000.0

# Some necessary functions:

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#def _bytes_feature_list(value):
#    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _Float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _Float_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def to_tf_feature(value):
    if type(value) == str:
        return _bytes_feature(tf.compat.as_bytes(value))
    
    elif type(value) is np.ndarray:
        dtype = value.dtype
        if dtype == np.float32:
            return _Float_feature_list(value)
        else:
            return _int64_feature_list(value)
        
    elif type(value) == int:
        return _int64_feature(value)

    elif type(value) == float or type(value) == np.float64:
        return _Float_feature(value)

    
    elif type(value) == list and type(value[0]) == str:
        return _bytes_feature(tf.compat.as_bytes(' '.join(value)))
        #return _bytes_feature_list(tf.compat.as_bytes(value))
    else:
        print( "Type of", value,"is", type(value)) 
        return _bytes_feature(value)
        
def load_audio( wav, config):
    wavs = []
    return wavs

def extract_and_save_to_record(writer = None,
                               sourcefile = '',
                               filepath = '',
                               lng = 'en_uk',
                               wordseq = '',
                               phoneseq = '',
                               classseq = [],
                               gender = 'f',
                               age = 30,
                               classes_framewise = [],
                               pdfs_framewise = [],
                               reduced_class_phones = '',
                               reduced_class_seq = [] ):
    

    if True:
        beginsample = 0
        endsample = -1
        with open(filepath, mode='rb') as file: # b is important -> binary
            file.seek(44 + int(beginsample) * bytes_per_sample ) # WAV header is 44 bytes - Hopefully always!
            if endsample > 0:                                   
                fileContent = file.read( (endsample-beginsample) * bytes_per_sample )
            else:
                fileContent = file.read()                    
                file.close()
                
                audio = np.array(unpack("h"* int(len(fileContent)/ bytes_per_sample), fileContent), dtype=np.int16).astype(np.float32)
                audio = audio / np.max(np.abs(audio))


    feature = {}
    feature['sourcefile'] = to_tf_feature( str(sourcefile) )
    feature['lng'] = to_tf_feature( str(lng) )
    feature['wordseq'] =  to_tf_feature( str(wordseq) )
    feature['phoneseq'] = to_tf_feature( phoneseq )
    feature['classseq'] = to_tf_feature( np.array(classseq).astype(int) )
    feature['gender'] = to_tf_feature( str(gender) )
    feature['age'] = to_tf_feature( int(age) )
    feature['audio'] = to_tf_feature(  audio )
    feature['classes_framewise'] = to_tf_feature(  classes_framewise )
    feature['pdfs_framewise'] = to_tf_feature(  pdfs_framewise )
    feature['phoneseq_reduced'] = to_tf_feature( reduced_class_phones )
    feature['classseq_reduced'] = to_tf_feature( reduced_class_seq )
    
    # Get openSMILE features:
    if True:
        pid = os.getpid()
        FNULL = open(os.devnull, 'w')
        smileoutputfile = '/tmp/SMILExtract_output_pid_%i.csv' % pid
        # run ./SMILExtract -C prosodyAcf.conf -I input.wav -O output.csv
        call( [opensmile_extractor,
               '-C',
               opensmile_config,
               '-I',
               filepath,
               '-O',
               smileoutputfile], stdout=FNULL, stderr=FNULL)
        
        # Opensmile features:
        # 0 frameIndex;
        # 1 frameTime;
        # 2 voiceProb_sma;
        # 3 F0_sma;
        # 4 pcm_intensity_sma;
        # 5 pcm_loudness_sma

        smilefeat = np.genfromtxt(smileoutputfile, delimiter=';', skip_header=1).astype(np.float32)

        _start = 0
        _end = len(pdfs_framewise)

        
        if len(pdfs_framewise) -  len(smilefeat[:,2]) > 1:
            _end += 1
            _start += 1 
            print("Kaldi alignments and smilefeats have different length %i < %i" % (len(pdfs_framewise),len(smilefeat[:,2]) ) )
        feature['voiceProb_framewise'] = to_tf_feature(  smilefeat[_start:_end,2] )
        feature['F0_framewise'] = to_tf_feature( smilefeat[_start:_end,3] )
        feature['intensity_framewise'] = to_tf_feature( smilefeat[_start:_end,4] )
        feature['loudness_framewise'] = to_tf_feature( smilefeat[_start:_end,5] )

    # Construct the Example proto object
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize the example to a string
    serialized = example.SerializeToString()
    
    # write the serialized objec to the disk
    writer.write(serialized)

    return 1
    
    
                               
def extract_and_save_to_record_old(utterance_writer = None,
                               phoneme_writers = [],
                               phoneme_writer_counters = [],
                               sourcefile = '',
                               filepath = '',
                               lng = 'en_uk',
                               wordseq = '',
                               phoneseq = [],
                               classseq = [],
                               starttime = 0,
                               duration = 1,
                               individual_phonemes = [],
                               gender = 'f',
                               age = 30,
                               endsillength = 0,
                               starttimeoffset = 0,
                               bucketlength = 1.0):


    if len(individual_phonemes) < 2 and individual_phonemes[0][2] == 'sil':
        print("Rejecting a single silence")
        return phoneme_writer_counters

    if len(individual_phonemes) == 2 and individual_phonemes[0][2] == 'sil' and individual_phonemes[1][2] == 'sil':
        print("Rejecting a double silence")
        return phoneme_writer_counters
       

    
    beginsample = starttime * fs 
    #if beginsample>0:
    #    print(sourcefile)
    #    print("Starttime:", starttime)
    #    print("Beginsample:", beginsample)
    endsample = -1
    #try:
    if True:
        with open(filepath, mode='rb') as file: # b is important -> binary
            file.seek(44 + int(beginsample) * bytes_per_sample ) # WAV header is 44 bytes - Hopefully always!
            if endsample > 0:                                   
                fileContent = file.read( (endsample-beginsample) * bytes_per_sample )
            else:
                fileContent = file.read()                    
                file.close()
                
                audio = np.array(unpack("h"* int(len(fileContent)/ bytes_per_sample), fileContent), dtype=np.int16).astype(np.float32)
                audio = audio / np.max(np.abs(audio))
    #except:
    #    print('Problem opening file %s' % filepath)
    #    return phoneme_writer_counters
    #print(np.array(classseq).astype(int))
    feature = {}
    feature['sourcefile'] = to_tf_feature( str(sourcefile) )
    feature['lng'] = to_tf_feature( str(lng) )
    feature['wordseq'] =  to_tf_feature( str(wordseq) )
    feature['phoneseq'] = to_tf_feature( phoneseq )
    feature['classseq'] = to_tf_feature( np.array(classseq).astype(int) )
    feature['starttime'] = to_tf_feature( float(starttime) )
    feature['duration'] = to_tf_feature( float(duration) )
    feature['gender'] = to_tf_feature( str(gender) )
    feature['age'] = to_tf_feature( int(age) )
    feature['endsillength'] = to_tf_feature( float(endsillength) )
    feature['bucketlength'] = to_tf_feature( float(bucketlength) )
    #for p in individual_phonemes:
    #    print(p)
    #print(np.array( [ s[:3] for s in individual_phonemes ]).flatten().astype(np.float32).reshape([-1,3]))
    feature['alignment'] = to_tf_feature( np.array( [ s[:3] for s in individual_phonemes ]).flatten().astype(np.float32)   ) 
    
    start = 0 #int(fs * (starttimeoffset + starttime))
    #end = int(fs*duration) #int(fs * (starttimeoffset + starttime + duration))
    
    #if beginsample>0:
    #print("individual_phonemes[-1] =", individual_phonemes[-1], "duration:", duration)
    end = int(individual_phonemes[-1][1] * fs)
    feature['audio'] = to_tf_feature(  audio[start:end] )

    # Construct the Example proto object
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize the example to a string
    serialized = example.SerializeToString()
    
    # write the serialized objec to the disk
    utterance_writer.write(serialized)

    
    for phoneme in individual_phonemes:
        feature = {}
        feature['sourcefile'] = to_tf_feature( str(sourcefile) )
        feature['starttime'] = to_tf_feature( float(phoneme[0]) )
        feature['endtime'] = to_tf_feature( float(phoneme[1]) )
        feature['class'] = to_tf_feature( int(phoneme[2]) )
        feature['phone'] = to_tf_feature( str(phoneme[3]) )
        feature['gender'] = to_tf_feature( str(gender) )
        feature['age'] = to_tf_feature( int(age) )

        start_padding =  int( min(fs * padding_s, start ))
        end_padding = int(min(fs * padding_s, len(audio) - (phoneme[1]) * fs  ))

        start = int( fs * (starttimeoffset + phoneme[0]) - start_padding)
        end = int( fs * (starttimeoffset +phoneme[1]) + end_padding)

        
        feature['start_padding_samples'] = to_tf_feature( start_padding )                                                          
        feature['end_padding_samples'] = to_tf_feature( end_padding )
        
        start= max(start,0)
        end=min(end,len(audio))

        #print(start,end)
        feature['audio'] = to_tf_feature( audio[start:end] )

        # Construct the Example proto object
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize the example to a string
        serialized = example.SerializeToString()

        writer = np.argmin(phoneme_writer_counters)
        # write the serialized objec to the disk
        phoneme_writers[writer].write(serialized)

        phoneme_writer_counters[writer] += 1
        
    return phoneme_writer_counters

        
# def write_to_tfrecord(writer, dataitem):
#     # Thanks to https://gist.github.com/swyoon/8185b3dcf08ec728fb22b99016dd533f
#     # iterate over each sample,
#     # and serialize it as ProtoBuf.

#     feature = {}
#     feature['nmic'] = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array([dataitem['microphones']])))
#     feature['y'] = tf.train.Feature(float_list=tf.train.FloatList(value=dataitem['labs'].astype(np.float32).flatten()))
#     feature['conf'] = tf.train.Feature(int64_list=tf.train.Int64List(value=dataitem['config']))
#     feature['src'] = tf.train.Feature(int64_list=tf.train.Int64List(value=dataitem['sourcefile']))

#     #print(dataitem['sourcefile'].decode(const.default_text_encoding))
#     #print(dataitem['config'].decode(const.default_text_encoding))
    
#     #import matplotlib.pyplot as plt

#     for n in range(4):
#         #plt.subplot(411+n)
#         if n < len(dataitem['wavs']):
#             feature['X%i' %n] = tf.train.Feature(float_list=tf.train.FloatList(value=dataitem['wavs'][n].astype(np.float32).flatten()))
#             #plt.plot(dataitem['wavs'][n])
#         else:
#             feature['X%i' %n] = tf.train.Feature(float_list=tf.train.FloatList(value=np.array([],dtype=np.float32).flatten()))

#     #plt.show()
            
#     # Construct the Example proto object
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
    
#     # Serialize the example to a string
#     serialized = example.SerializeToString()
    
#     # write the serialized objec to the disk
#     writer.write(serialized)
    
def mkdir(path):
    try:
        os.makedirs(path)        
    except OSError as exc:  # Python >2.5
        #print ("dir %s exists" % path)
        dummy = 1

    
# def load_or_create_tfrecord( dataset=None, corpora=None, corpusconfig=None, featureconfig=None,data_cache_dir='/tmp/' ):

#     writers = []
#     for shard in range(const.num_tfrecord_shards):
#         targetfile = os.path.join(data_cache_dir, 'collection.%s.tfrecord.%i' % ( corpus, shard ))
#         writers.append (tf.python_io.TFRecordWriter(targetfile))
#         counts.append(0)

#     print("Let's start packing this into a record")
#     sys.stdout.write('%08i'% 0)
#     sys.stdout.flush()

#     permut = np.random.permutation(int(np.max(labs[:, const.filenameid_col])+1))
#     for i in range(int(np.max(labs[:, const.filenameid_col])+1)):
#                 #print(i, permut[i])
                
#                 lab_ind =  np.where(labs[:, const.filenameid_col] == permut[i])[0]
#                 segmentinfo = labs[lab_ind,:]

#                 all_wavs = feature_extractor.extract_complete_utterance_for_all_mics(corpusconfigs,
#                                                                                      featureconfig,
#                                                                                      segmentinfo,
#                                                                                      wavs[ int(permut[i]) ])


#                 if len(all_wavs[0]) > 2:
#                     dataitem = {}
#                     dataitem['wavs'] = all_wavs
#                     dataitem['labs'] = segmentinfo
#                     dataitem['microphones'] = len(all_wavs)
#                     dataitem['config'] = corpus.encode(const.default_text_encoding) #[ord(c) for c in corpus]
#                     dataitem['sourcefile'] = wavs[ int(permut[i]) ].encode(const.default_text_encoding)
#                     #print (wavs[ int(i) ])

#                     write_to_tfrecord(writers[i%const.num_tfrecord_shards], dataitem)
#                     counts[i%const.num_tfrecord_shards] += 1
#                 if i % 10 == 0:
#                     sys.stdout.write('\b\b\b\b\b\b\b\b%08i'%i)
#                     sys.stdout.flush()


#     for writer in writers:
#         writer.close()
#     print("... Done")


#     for shard in range(const.num_tfrecord_shards):
#         countfile = os.path.join(data_cache_dir, 'collection.%s.itemcount.%i' % ( corpus, shard ))
#         np.savetxt(countfile, [counts[shard]], fmt='%i')

            
#     return targetfiles, configs, counts





# Load phone definitions:

class_to_phone = {}
phone_to_class = {}


for l in open(phonesetfile, 'r').readlines():
    ph,cl = l.strip().split()
    class_to_phone[cl] = ph
    phone_to_class[ph] = cl


kaldi_class_to_class = {}
kaldi_phone_to_short_phone = {}
short_phone_to_kaldi_phone = {}
short_phone_to_class = {'_' : 0}



for k in phone_to_class.keys():
    if '#' not in k:
        kaldi_phone_to_short_phone[k] = re.sub(r'_[BEIS]', '',k)
        short_phone_to_kaldi_phone[re.sub(r'_[BEIS]', '',k)] = k

ct = 3
for k in short_phone_to_kaldi_phone.keys():
 if k == 'sil' or k == '<eps>':
  short_phone_to_class[k] = 0
 elif  k == 'spn' or k == 'spn_S':
  short_phone_to_class[k] = 1
 elif k == 'nsn':
  short_phone_to_class[k] = 2
 else:
  short_phone_to_class[k] = ct
  ct += 1

class_to_class = {}
for k,p in class_to_phone.items():
    p = re.sub(r'_[BEIS]', '',p)
    if p in short_phone_to_class.keys():
        class_to_class[int(k)] = short_phone_to_class[re.sub(r'_[BEIS]', '',p)]
        

for phone in duration_cutoffs.keys():
    short_phone = phone.split('_')[0]
    if short_phone+lengthening_mark not in short_phone_to_class.keys():
        short_phone_to_class[short_phone+lengthening_mark] = ct
        ct += 1
    next_kaldi_class = len(class_to_class.keys())
    class_to_class[next_kaldi_class] = short_phone_to_class[short_phone+lengthening_mark]
    phone_to_class[phone.replace('_', lengthening_mark + '_')] = next_kaldi_class


mkdir(data_cache_dir)


if par_num == 0:
    phonesoutfile = open(os.path.join(data_cache_dir, 'phones.conf'), 'w')


    print("Writing phone & class info to %s" % phonesoutfile)

    phonesoutfile.write('[kaldi_class_to_class]\n')

    for k,v in class_to_class.items():
        phonesoutfile.write("%i = %i\n" % (int(k), int(v)))

    phonesoutfile.write('\n\n')
    phonesoutfile.write('[phoneme_class]\n')

    for k,v in short_phone_to_class.items():
        phonesoutfile.write("%s = %i\n" % (k, int(v)))


    articulatory_features = {}
    phonedefs = {}
    phonecategories = {}
    for l in ipadefinitions.keys():
        for k,p in ipadefinitions[l].items():
            if p['ipa'] not in phonedefs.keys():
                phonedefs[p['ipa']] = {'features' : p['features'],
                                       'category' : p['category'].lower() }
            if p['category'].lower() not in phonecategories.keys():
                phonecategories[p['category'].lower()] = {'phones' : []}
            phonecategories[p['category'].lower()]['phones'].append(p['ipa'])

    phonesoutfile.write('\n\n')
    phonesoutfile.write('[phoneme_to_articulatory]\n')

    unfortunate_phone_map = {}

    for k in short_phone_to_class.keys():
        if k in phonedefs.keys():
            feat = phonedefs[k]['features']
            for f in feat:
                articulatory_features[f] = 1
            phonesoutfile.write("%s = %s\n" % (k, ','.join(feat)))
            unfortunate_phone_map[k] = k

        elif k.replace(lengthening_mark, '') in phonedefs.keys():
            print("%s shortened" % k)
            print("What to do with phone %s" % k)
            feat = ','.join(phonedefs[ k.replace(lengthening_mark, '') ]['features'])
            for f in feat:
                articulatory_features[f] = 1
            phonesoutfile.write("%s = %s\n" % (k, ','.join(feat)))
            unfortunate_phone_map[k] = k.replace(lengthening_mark, '')

        elif k + lengthening_mark in phonedefs.keys():
            print("%s lenghtened" % k)
            feat = phonedefs[ k + lengthening_mark ]['features']
            for f in ['long', 'geminated']:
                if f in feat:
                    feat.remove(f)
            #feat = re.sub(',long|long,', '', feat)
            #feat = re.sub(',geminated|geminated,', '', feat)
            for f in feat:
                articulatory_features[f] = 1
            phonesoutfile.write("%s = %s\n" % (k, ','.join(feat)))
            unfortunate_phone_map[k] = k + lengthening_mark
        else:
            print("Trouble: No mention of phone %s in definitions" % k)


    phonesoutfile.write('\n\n')
    phonesoutfile.write('[articulatory_class]\n')
    art_counter = 0
    for k in articulatory_features.keys():
        phonesoutfile.write("%s = %i\n" % (k, art_counter))
        art_counter += 1



    phonesoutfile.write('\n\n')
    phonesoutfile.write('[event_to_class]\n')

    most_popular_phones = {}

    catct = 0        
    for cat in phonecategories.keys():
        categoryphones,counts = np.unique(phonecategories[cat]['phones'], return_counts=True)
        most_popular_phones[cat] = categoryphones[np.argmax(counts)]
        category_index = catct
        catct += 1
        phonesoutfile.write("%s = %s\n" % (cat, category_index))

    phonesoutfile.write('\n\n')
    phonesoutfile.write('[event_representations]\n')

    for cat in phonecategories.keys():
        phonesoutfile.write("%s = %s\n" % (cat, most_popular_phones[cat]))



    phonesoutfile.write('\n\n')
    phonesoutfile.write('[phoneme_to_event]\n')  

    for k in short_phone_to_class.keys():
        if k in phonedefs.keys():
            phonesoutfile.write("%s = %s\n" % (k, phonedefs[k]['category']))
        elif k.replace(lengthening_mark, '') in phonedefs.keys():
            phonesoutfile.write("%s = %s\n" % (k, phonedefs[ k.replace(lengthening_mark, '') ]['category']))
        elif k + lengthening_mark in phonedefs.keys():
            phonesoutfile.write("%s = %s\n" % (k, phonedefs[ k + lengthening_mark ]['category']))
        else:
            print("Trouble: No mention of phone %s in definitions" % k)


    
        
# Create alignment ctm files:


for dataset in ['train', 'devel', 'test', 'siak', 'tellme']:
    #for alignmentset in alignmentdirs[dataset]:
    for setnumber in range(len(alignmentdirs[dataset])):
        if par_num != setnumber % num_parallel:
            continue
        
        alignmentset = alignmentdirs[dataset][setnumber]
        alignmentdir = os.path.join(alignmentbasedir, alignmentset )


        utterance_record_dir = os.path.join(data_cache_dir, 'utterances', dataset)
        phoneme_record_dir =  os.path.join(data_cache_dir, 'phonemes', dataset)

        mkdir(utterance_record_dir)
        mkdir(phoneme_record_dir)
        
        promptsfile = os.path.join(labelbasedir, alignmentset.replace('tri3b_ali_', ''),  'text')
        filescpfile = os.path.join(labelbasedir, alignmentset.replace('tri3b_ali_', ''),  'wav.scp')

        prompts={}
        print("Reading prompts from %s" % promptsfile)
        for l in open(promptsfile,'r').readlines():
            f,p = l.strip().split(' ',1)
            prompts[f] = p
            
        filescp={}
        print("Reading filepaths from %s" % filescpfile)
        for l in open(filescpfile,'r').readlines():
            f,p = l.strip().split(' ',1)    
            filescp[f] = p

        
        i = 1

        print("Checking for file %s/ali.%i.gz" % ( alignmentdir, i))
        while os.path.exists("%s/ali.%i.gz" % ( alignmentdir, i) ):
            outfile='/tmp/%s_%s_ali_for_tfrecords.%i' %(kaldidir, alignmentset, i)
            if os.path.exists(outfile):
                print("File %s exists already" % outfile)
            else:
                print("Outputting phone alignments to %s" % outfile)
                f = open(outfile, 'w')
                call( [ali_to_phones,
                       #'--ctm-output',
                       '--per-frame',
                       modelfile,
                       'ark:gunzip -c %s/ali.%i.gz |' % (alignmentdir, i),
                       'ark,t:-'], stdout=f)

            outfile='/tmp/%s_%s_pdf_for_tfrecords.%i' %(kaldidir, alignmentset, i)
            if os.path.exists(outfile):
                print("File %s exists already" % outfile)
            else:
                print("Outputting PDF alignments to %s" % outfile)
                f = open(outfile, 'w')
                call( [ali_to_pdfs,
                       #'--ctm-output',
                       modelfile,
                       'ark:gunzip -c %s/ali.%i.gz |' % (alignmentdir, i),
                       'ark,t:-'], stdout=f)

            i+=1
            numfiles = i
 
        files_to_data = {}
        plen_stats = []
        ct = 0
        for i in range(1, numfiles):
            print("Processing phone ali-file %i" % i)
            
            for l in open('/tmp/%s_%s_ali_for_tfrecords.%i' %(kaldidir, alignmentset, i), 'r').readlines():
                filename_and_phones = l.strip().split()
                f = filename_and_phones[0]
                phones_framewise = filename_and_phones[1:]
                phones = [k for k,g in groupby(phones_framewise) if k!=0]
                plen = len(phones)
                plen_stats.append(plen)
                
                flen = len(phones_framewise)
                #print(plen)

                files_to_data[f] = { 'phones_framewise' : phones_framewise,
                                     'phones_seq' : phones,
                                     'phone_len' : plen,
                                     'frame_len' : flen,
                                     'length_key' : int(10000000 * plen + flen * 10000 + ct),
                                     'sourcefile' : f }
                ct += 1
        for i in range(1, numfiles):
            print("Processing pdf ali-file %i" % i)
            ct = 0
            for l in open('/tmp/%s_%s_pdf_for_tfrecords.%i' %(kaldidir, alignmentset, i), 'r').readlines():
                filename_and_phones = l.strip().split()
                f = filename_and_phones[0]
                pdfs_framewise = filename_and_phones[1:]
                flen=len(pdfs_framewise)

                if flen != files_to_data[f]['frame_len']:
                    print("Trouble, mismatching pdf and phone frame lengths %i, %i in %s" % (flen, files_to_data[f]['frame_len'] ,f))
                files_to_data[f]['pdfs_framewise'] = pdfs_framewise
                
        #splits = np.array_split(sorted(plen_stats), len(plen_stats)//approx_num_items_in_record + 1)
        #print([s[-1] for s in splits])
        
        #newlist = sorted(files_to_data.values(), key=lambda k: k['length_key'])
        if plotting:
            [counts, binedges, paches] = plt.hist(plen_stats, bins=100)
            plt.hist(plen_stats, bins=100)

        length_sorted_lists = np.array_split( sorted(files_to_data.values(), key=lambda k: k['length_key']), len(plen_stats)//approx_num_items_in_record + 1)

        for items in length_sorted_lists:
            
            uttctfile = os.path.join(utterance_record_dir, '%s.utterance_collection.%i-%i.counter' % ( alignmentset, items[0]['phone_len'], items[-1]['phone_len'] ))
            print('checking %s' % uttctfile)

            if os.path.exists( uttctfile ):
                uttminct = np.loadtxt(uttctfile)
            else:
                uttminct = 0

            if uttminct > 0:
                print(uttctfile, 'already done')
                continue
            
            uttrecordfile = os.path.join(utterance_record_dir, '%s.utterance_collection.%i-%i.tfr' % ( alignmentset, items[0]['phone_len'], items[-1]['phone_len'] ))
            
            print("%i candidates, let's put them into %s" % (len(items), uttrecordfile ))

            for r in range(80):
                print("-", end='')
            print('')
            
            if tf.__version__[0] == '1':
                writer = tf.python_io.TFRecordWriter(uttrecordfile)
            else:
                writer = tf.compat.v1.python_io.TFRecordWriter(uttrecordfile)
            np.savetxt(uttctfile, [-1], fmt='%i')
            cnt = 0
            
            for d in items:
                f = d['sourcefile']
                if f not in prompts.keys():
                    print("No prompt found so rejecting %s" % f)
                    continue
                
                wseq = prompts[f]
                if '<UNK>' in wseq:
                    print ("<UNK> in prompt of %s" % f)
                    continue      
                
                lngo = re.search('..\_..\_', wseq)
                if lngo is not None:
                    lng = lngo.group(0)[:-1]
                else:
                    lng = 'unk__'
                wseq = wseq.replace(lng+'_','')
                
                src =  f
                if src in filescp.keys():
                    filepath = filescp[ src ]
                elif src[:-2] in filescp.keys():
                    filepath = filescp[ src[:-2] ]
                else:
                    print("Could not find sourcefile %s" % src)
                    continue
                
                gender = f[40] # 'm' #genders[k][j]
                if gender.lower() != 'm' and gender.lower() != 'f':
                    print ("Gender '%s' not m or f for %s" % (gender, f))
                    continue

                try:
                    age = int(f[41:43]) #int(ages[k][j])
                except ValueError:
                    print ("Age '%s' not an integer for %s" % (ages[k][j],f))
                    continue

                #individual_phonemes = d['phones_seq']

                cseq = d['phones_seq'] #classseqs[k][j]
                pseq =  [class_to_phone[c] for c in cseq]
            
                pstring = ' '.join(pseq)
                pstring = re.sub('sil', '_', pstring)
                pstring = re.sub('spn_S', '_', pstring)
                pstring = re.sub('spn', '_', pstring)
                pstring = re.sub('nsn', '_', pstring)                
                pstring = re.sub('\_(S|E)', ' _', pstring)
                pstring = re.sub('\_(B|I)', '', pstring)
                pstring = re.sub('(\_ )+', '_ ', pstring)
                pstring = re.sub('( \_)+', ' _', pstring)                
                pstring = re.sub('(\_\_+)+', '_', pstring)
                #pstring

                reduced_class_seq = np.array([class_to_class[int(c)] for c in cseq], dtype=np.int64)

                #foo()
                
                phoneme_counts = extract_and_save_to_record( writer = writer,
                                                             sourcefile = f,
                                                             filepath = filepath,
                                                             lng = lng,
                                                             wordseq = wseq,
                                                             phoneseq = ' '.join(pseq),
                                                             classseq = cseq,
                                                             reduced_class_phones = pstring,
                                                             reduced_class_seq = reduced_class_seq,
                                                             gender = gender,
                                                             age = age,
                                                             classes_framewise = d['phones_framewise'],
                                                             pdfs_framewise = d['pdfs_framewise'],)

                
                cnt += 1
                if cnt % (len(items) // 80) == 0 :
                    print("#", end = '')
                    sys.stdout.flush()
                #  break
            np.savetxt(uttctfile, [cnt], fmt='%i')
            print("\n%i utterance segments processed" % cnt)            

            
        #values, bin_edges = np.histogram(np.array(plen_stats), bins=100, range=[0,100])
        #print(values)
        #print(bin_edges)
            
        '''    
            for l in open('/tmp/%s_%s_ali_for_tfrecords.%i' %(kaldidir, alignmentset, i), 'r').readlines():

                f,_,s,d,p = l.strip().split()
                if debug:
                    print(f,s,d,class_to_phone[p])

                #if (phone == 'sil' and float(d) > 0.4) or f != oldf:
                if f != oldf or (split_files and oldphone == 'sil' and float(oldd) > 1.0 and plen>1 ):
                    #print("f is not oldf")
                    if oldf is not '':
                            sourcefiles[corpus].append(oldf)
                            if not split_files:
                                starttimes[corpus].append(0)
                            else:
                                starttimes[corpus].append(begintime)
                                if f == oldf:
                                    #print("mid sentence split %s" % f)
                                    ind_ph[-1][1] = 0.5 * ind_ph[-1][1]
                                    print("Appending to phoneseqs et al again!")
                                    phoneseqs[corpus].append(pseq)
                                    classseqs[corpus].append(cseq)
                                else:
                                    print("I would append to phoneseqs but f %s != oldf %s!"%(f, oldf))
                            if oldphone == 'sil':
                                endsilencelengths[corpus].append(float(oldd))
                                durations[corpus].append(float(olde) + float(oldd) - begintime)
                            else:
                                endsilencelengths[corpus].append( 0.0 )
                                durations[corpus].append(float(olde) + float(oldd) - begintime)

                            ages[corpus].append(age)
                            genders[corpus].append(gender)
                            phonemes_for_files[corpus].append(ind_ph)
                            phoneseqs[corpus].append(pseq)
                            classseqs[corpus].append(cseq)

                    else:
                        print("oldf is ")
                    ind_ph = [] # [float(s),float(d),int(p), phone] ]
                    pseq = []
                    cseq = []
                    plen = 0
                    
                    #pseq = [phone]
                    #cseq = [int(p)]
                    #plen=1

                    age=f[41:43]
                    if age == '--':
                        print("Age problem: %s %s"%(age, f))
                    gender=f[40]

                    #if phone == 'sil':                
                    #    begintime = float(s) #+ 0.5 * float(d)
                    #else:
                    #    begintime = float(s)
                    if f == oldf:
                        print("Problem! f == oldf")
                        sys.exit(1)
                        begintime = float(olds) + 0.5 * float(oldd)
                    else:
                        begintime = 0
                        
                    if debug:
                        print('begintime =',begintime)
                    corpus=f[:40]
                    if corpus not in corpora:
                        print("corpus %s not in"%corpus, corpora) 
                        #chopped_phone_lengths[corpus] = []
                        #chopped_time_lengths[corpus] = []

                        sourcefiles[corpus] = []
                        starttimes[corpus] = []
                        durations[corpus] = []
                        phoneseqs[corpus] = []
                        classseqs[corpus] = []
                        wordseqs[corpus] = []
                        ages[corpus] = []
                        genders[corpus] = []
                        endsilencelengths[corpus] = []
                        phonemes_for_files[corpus] = []

                        corpora.append(corpus)
                    ct += 1

                phone = class_to_phone[p]
                if phone in duration_cutoffs.keys() and float(d) > duration_cutoffs[phone]/100:
                    phone = phone.replace('_', lengthening_mark + '_')
                    #print("Lengthening %s => %s for duration %0.2f" % (class_to_phone[p], phone, float(d)))                    
                    p = phone_to_class[phone]
                #print("Appending phone.split('_')[0]", phone.split('_')[0])
                pseq.append( phone.split('_')[0] )                    
                cseq.append( int(p) )
                ind_ph.append([float(s)-begintime,float(s)+float(d)-begintime,int(p), phone])
                plen += 1
                
                oldf = f    
                olde = float(s)+float(d) # start + duration
                oldd = float(d)
                olds = float(s)
                oldphone = phone

            if plen > 1:
                print("Adding to phoneseqs et al")
                sourcefiles[corpus].append(oldf)
                starttimes[corpus].append(begintime)
                phoneseqs[corpus].append(pseq)
                classseqs[corpus].append(cseq)
                ages[corpus].append(age)
                genders[corpus].append(gender)
                if phone == 'sil':
                    endsilencelengths[corpus].append(float(d))
                    durations[corpus].append(float(olde) + float(d) - begintime)
                else:
                    endsilencelengths[corpus].append(0.0)
                    durations[corpus].append(float(olde) + float(d) - begintime)
                phonemes_for_files[corpus].append(ind_ph)
            #break
        '''

        #for i in np.argsort(np.array( durations )):
        #    print(sourcefiles[i],starttimes[i], durations[i], phoneseqs[i])

        #for c in phone_lengths.keys():
        #    print(c)
        #    #print(np.bincount(phone_lengths[c]))
        #    print(np.bincount(chopped_phone_lengths[c]))
        #    print(np.histogram(chopped_time_lengths[c],np.arange(0,4,0.25)))


        #for k in durations.keys():
        #    durations[k]=np.array(durations[k])

        # selected_corpora = ['pfstar_enuk____train_clean_wav__________',
        #                     'speechdat-fi___train_tele_FIA___________',
        #                     'speecon-fi_____train_clean_FI0__________',
        #                     'speecon-fi_____train_noisy_FI0__________',
        #                     'speeconkids-fi_train_clean_FI0__________',
        #                     'spraakbanken_setrain_random17k_wav______',
        #                     'tidigits_______children_train_wav_______',
        #                     'wsj1___________wsj1_si_tr_s_wv1_________',
        #                     'wsjcam0________si_tr_clean_wv1__________']

        # selected_mics = ['wav', 'wv1', 'wav']


        # for k in corpora:
        #   if 'FI2' not in k and 'FI3' not in k:
              
        #       #for i in range(len(phonecountbins)):
                  
        #       #timebin = timebins[i]
        #       phonecountbin = phonecountbins[i]
             
        #       recordfiles = []

        #       #n = len(np.intersect1d(np.where(np.array(durations[k]) > timebin[0])[0], np.where(np.array(durations[k]) <= timebin[1])[0]))
        #       n = len(np.intersect1d(np.where(np.array(phonecounts[k]) > phonecountbin[0])[0], np.where(np.array(phonecounts[k]) <= phonecountbin[1])[0]))
        #       recordfilecount=int(np.ceil(n / utts_per_record ))

        #       if recordfilecount == 0:
        #           continue

        #       utterance_writers = []
        #       phoneme_writers = []

        #       utterance_counts = []
        #       phoneme_counts = []

        #       utterance_counter_files = []
        #       phoneme_counter_files = []

        #       uttminct = np.inf
        #       phminct = np.inf
        #       for shard in range(recordfilecount):
        #           uttctfile = os.path.join(utterance_record_dir, 'utterance_collection.%s.%0.1f-%0.1fs.counter.%i' % ( k, timebin[0], timebin[1], shard ))
        #           print('checking %s' % uttctfile)
        #           utterance_counter_files.append(uttctfile )

        #           if os.path.exists( uttctfile ):
        #               uttminct = min(uttminct, np.loadtxt(uttctfile).item())
        #           else:
        #               uttminct = 0

        #           phctfile = os.path.join(phoneme_record_dir, 'phoneme_collection.%s.%0.1f-%0.1fs.counter.%i' % ( k, timebin[0], timebin[1], shard ))

        #           print('checking %s' % phctfile)
        #           phoneme_counter_files.append( phctfile )          


        #           if os.path.exists( phctfile ):
        #               phminct = min(phminct, np.loadtxt(phctfile).item())
        #           else:
        #               phminct = 0

        #           print("utt min %i ph min %i" % (uttminct,phminct))            

        #       if uttminct > 0 and phminct > 0:
        #           print('utterance_collection.%s.%0.2fs.tfrecord.%i' % ( k, timebin[1], shard ), 'already done')
        #           print('phoneme_collection.%s.%0.2fs.tfrecord.%i' % ( k, timebin[1], shard ), 'already done')
        #           continue

        #       print("%i candidates, put into following files:" % n)              
        #       for shard in range(recordfilecount): 
        #           print("  Shard", shard, "of", recordfilecount)
        #           utterance_targetfile = os.path.join(utterance_record_dir, 'utterance_collection.%s.%0.1f-%0.1fs.tfrecord.%i' % ( k, timebin[0], timebin[1], shard ))
        #           utterance_writers.append (tf.python_io.TFRecordWriter(utterance_targetfile))
        #           phoneme_targetfile = os.path.join(phoneme_record_dir, 'phoneme_collection.%s.%0.1f-%0.1fs.tfrecord.%i' % ( k, timebin[0], timebin[1], shard ))
        #           phoneme_writers.append (tf.python_io.TFRecordWriter(phoneme_targetfile))
        #           utterance_counts.append(0)
        #           phoneme_counts.append(0)

        #           np.savetxt(phoneme_counter_files[shard], [-1], fmt='%i')
        #           np.savetxt(utterance_counter_files[shard], [-1], fmt='%i')

        #           print("  ",utterance_targetfile)
        #           print("  ",phoneme_targetfile)
        #       cnt = 0

        #       for j in np.random.permutation(np.intersect1d(np.where(np.array(durations[k]) > timebin[0])[0], np.where(np.array(durations[k]) <= timebin[1])[0])):
        #         f = sourcefiles[k][j]

                
                
        #         if f not in prompts.keys():
        #             print("No prompt found so rejecting %s" % f)
        #             continue
        #         wseq = prompts[f]
        #         if '<UNK>' in wseq:
        #             print ("<UNK> in prompt of %s" % f)
        #             continue      

        #         lngo = re.search('..\_..\_', wseq)
        #         if lngo is None:
        #             continue
        #         lng = lngo.group(0)[:-1]
        #         wseq = wseq.replace(lng+'_','')

        #         src =  sourcefiles[k][j]
        #         if src in filescp.keys():
        #             filepath = filescp[ src ]
        #         elif src[:-2] in filescp.keys():
        #             filepath = filescp[ src[:-2] ]
        #         else:
        #             print("Could not find sourcefile %s" % src)
        #             continue
        #         starttime = starttimes[k][j]
        #         duration = durations[k][j]
        #         endsil = endsilencelengths[k][j]

        #         startoffset = 0 #starttimescp[f]
        #         if startoffset != 0:
        #             print("File %s start offset %f" % (f, startoffset))

        #         gender = 'm' #genders[k][j]
        #         if gender.lower() != 'm' and gender.lower() != 'f':
        #             print ("Gender '%s' not m or f for %s" % (gender, f))
        #             continue

        #         try:
        #             age = int(ages[k][j])
        #         except ValueError:
        #             print ("Age '%s' not an integer for %s" % (ages[k][j],f))
        #             continue

        #         #if j not in phonemes_for_files[k].keys():
        #         #    print("Trouble: No individual phonemes for index %i file %s" % (j,f))
        #         #    continue
        #         individual_phonemes = phonemes_for_files[k][j]

        #         #print( k, '\t', j,'\t', f)                
        #         #print("Number of phoneseqs for corpus %s:"%k,len(phoneseqs[k]))
        #         #print(phoneseqs[k])
        #         pseq = phoneseqs[k][j]
        #         if ' '.join(pseq) == 'sil s p n sil':
        #             continue

        #         cseq = classseqs[k][j]

        #         writer = np.argmin(utterance_counts)
        #         phoneme_counts = extract_and_save_to_record(utterance_writer = utterance_writers[writer],
        #                                                     phoneme_writers = phoneme_writers,
        #                                                     phoneme_writer_counters = phoneme_counts,
        #                                                     sourcefile = f,
        #                                                     filepath = filepath,
        #                                                     lng = lng,
        #                                                     wordseq = wseq,
        #                                                     phoneseq = pseq,
        #                                                     classseq = cseq,
        #                                                     starttime = starttime,
        #                                                     duration = duration,
        #                                                     individual_phonemes = individual_phonemes,
        #                                                     gender = gender,
        #                                                     age = age,
        #                                                     endsillength = endsil,
        #                                                     starttimeoffset = startoffset,
        #                                                     bucketlength=timebin[1]

        #         )
        #         #print(filescp[ [sourcefiles[k][j]] ],timebin[0],'<',"%0.2f" %durations[k][j],'<',timebin[1], phoneseqs[k][j])
        #         utterance_counts[writer] += 1
        #         cnt += 1
        #         #if cnt>10:
        #         #  break
        #       for shard in range(len(utterance_counter_files)):
        #           np.savetxt(utterance_counter_files[shard], [utterance_counts[shard]], fmt='%i')
        #           np.savetxt(phoneme_counter_files[shard], [phoneme_counts[shard]], fmt='%i')
        #       print("%i utterance segments processed" % cnt)

