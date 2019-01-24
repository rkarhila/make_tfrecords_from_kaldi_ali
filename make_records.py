import os
from subprocess import call
import numpy as np

import tensorflow as tf
import re

from struct import unpack

debug = False

data_cache_dir = '/teamwork/t40511_asr/scratch/rkarhila/corpora_for_new_siak/new_new_new_tfrecords/'

phonesetfile='/l/rkarhila/kaldi-trunk/egs/multicorpora_all_corpora.old/s5/data/lang/phones.txt'

labelbasedir='/l/rkarhila/kaldi-trunk/egs/multicorpora_all_corpora.old/s5/data'
modelfile='/l/rkarhila/kaldi-trunk/egs/multicorpora_all_corpora.old/s5/exp/tri3b/final.mdl'

promptsfile = '/teamwork/t40511_asr/scratch/rkarhila/corpora_for_new_siak/kaldi_necessities/all_corpora_all_files/data/text'
filescpfile = '/teamwork/t40511_asr/scratch/rkarhila/corpora_for_new_siak/kaldi_necessities/all_corpora_all_files/data/wav.scp'

alignmentbasedir='/l/rkarhila/kaldi-trunk/egs/multicorpora_all_corpora.old/s5/exp/'

#modelfile='/l/rkarhila/kaldi-trunk/egs/multicorpora_all_corpora.old/s5/exp/tri3b/final.mdl'

#alignmentdir='/l/rkarhila/kaldi-trunk/egs/multicorpora_all_corpora.old/s5/exp/tri3b_ali_sp_cleaned_train/'

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
                           "tri3b_ali_devel_wsjcam0_si_dt_1_clean_wv2"],
                'test' : [ "tri3b_ali_test_tidigits_children",               
                           #"tri3b_ali_cleaned_train",
                           #"tri3b_ali_train",
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
                           "tri3b_ali_test_tidigits_children" ] }

speaker_age_file=''

ali_to_phones='/l/rkarhila/kaldi-trunk/src/bin/ali-to-phones'
#ali_to_phones='ali-to-phones'

timebins = [ [0.1, 2.0],
             [2.0, 4.0],
             [4.0, 6.0],
             [6.0, 8.0],
             [8.0, 10.0],
             [10.0, 12.0],
             [12.0, 18.0],
             [18.0, 30.0] ]

padding_s = 0.04
fs = 16000
bytes_per_sample = 2

utts_per_record = 3000.0

# Some necessary functions:

def load_waveform_chunk(filepath, beginsample=0, endsample=-1, bytes_per_sample=2):

    #print ("bytes per sample: ", bytes_per_sample)
    try:
        with open(filepath, mode='rb') as file: # b is important -> binary
            file.seek(44 + beginsample * bytes_per_sample ) # WAV header is 44 bytes - Hopefully always!
            if endsample > 0:                                   
                fileContent = file.read( (endsample-beginsample) * bytes_per_sample )
            else:
                fileContent = file.read()                    
                file.close()
                
                raw_audio_chunk = np.array(unpack("h"* int(len(fileContent)/ bytes_per_sample), fileContent), dtype=np.float)
    except:
        return None

    # Here's a little annoying thing to be figured out later:
    if len(raw_audio_chunk) == 0:
        print("Problem! Start" , segmentinfo[1], "->", trimstart, "End" , segmentinfo[2], "->", trimend, "File",filepath )
        #return np.zeros(0, dtype=np.float32)
        return np.zeros((0,featuredim), dtype=np.float32), None
    #else:
    #    print("No problem:  Start" , segmentinfo[1], "->", trimstart, "End" , segmentinfo[2], "->", trimend, "File",filepath )
    
    # 2 Normalise and select correct chunk:
    if np.max(raw_audio_chunk[:]) > 0:
        raw_audio_chunk -= np.mean(raw_audio_chunk)
        raw_audio_chunk /= np.max(raw_audio_chunk[:])
    else:
        #return np.zeros(0, dtype=np.float32)
        return [], None #np.zeros((0,featuredim), dtype=np.float32)
    
    #raw_audio_chunk = raw_audio_chunk[ beginsample : endsample ]
    return raw_audio_chunk


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


def extract_and_save_to_record(utterance_writer = None,
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

    beginsample = 0 #starttime * fs * bytes_per_sample  
    endsample = -1
    #try:
    if True:
        with open(filepath, mode='rb') as file: # b is important -> binary
            file.seek(44 + beginsample * bytes_per_sample ) # WAV header is 44 bytes - Hopefully always!
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
    
    start = int(fs * (starttimeoffset + starttime))
    end = int(fs * (starttimeoffset + starttime + duration))
    
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

        
def write_to_tfrecord(writer, dataitem):
    # Thanks to https://gist.github.com/swyoon/8185b3dcf08ec728fb22b99016dd533f
    # iterate over each sample,
    # and serialize it as ProtoBuf.

    feature = {}
    feature['nmic'] = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array([dataitem['microphones']])))
    feature['y'] = tf.train.Feature(float_list=tf.train.FloatList(value=dataitem['labs'].astype(np.float32).flatten()))
    feature['conf'] = tf.train.Feature(int64_list=tf.train.Int64List(value=dataitem['config']))
    feature['src'] = tf.train.Feature(int64_list=tf.train.Int64List(value=dataitem['sourcefile']))

    #print(dataitem['sourcefile'].decode(const.default_text_encoding))
    #print(dataitem['config'].decode(const.default_text_encoding))
    
    #import matplotlib.pyplot as plt

    for n in range(4):
        #plt.subplot(411+n)
        if n < len(dataitem['wavs']):
            feature['X%i' %n] = tf.train.Feature(float_list=tf.train.FloatList(value=dataitem['wavs'][n].astype(np.float32).flatten()))
            #plt.plot(dataitem['wavs'][n])
        else:
            feature['X%i' %n] = tf.train.Feature(float_list=tf.train.FloatList(value=np.array([],dtype=np.float32).flatten()))

    #plt.show()
            
    # Construct the Example proto object
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize the example to a string
    serialized = example.SerializeToString()
    
    # write the serialized objec to the disk
    writer.write(serialized)
    
def mkdir(path):
    try:
        os.makedirs(path)        
    except OSError as exc:  # Python >2.5
        #print ("dir %s exists" % path)
        dummy = 1

    
def load_or_create_tfrecord( dataset=None, corpora=None, corpusconfig=None, featureconfig=None,data_cache_dir='/tmp/' ):

    writers = []
    for shard in range(const.num_tfrecord_shards):
        targetfile = os.path.join(data_cache_dir, 'collection.%s.tfrecord.%i' % ( corpus, shard ))
        writers.append (tf.python_io.TFRecordWriter(targetfile))
        counts.append(0)

    print("Let's start packing this into a record")
    sys.stdout.write('%08i'% 0)
    sys.stdout.flush()

    permut = np.random.permutation(int(np.max(labs[:, const.filenameid_col])+1))
    for i in range(int(np.max(labs[:, const.filenameid_col])+1)):
                #print(i, permut[i])
                
                lab_ind =  np.where(labs[:, const.filenameid_col] == permut[i])[0]
                segmentinfo = labs[lab_ind,:]

                all_wavs = feature_extractor.extract_complete_utterance_for_all_mics(corpusconfigs,
                                                                                     featureconfig,
                                                                                     segmentinfo,
                                                                                     wavs[ int(permut[i]) ])


                if len(all_wavs[0]) > 2:
                    dataitem = {}
                    dataitem['wavs'] = all_wavs
                    dataitem['labs'] = segmentinfo
                    dataitem['microphones'] = len(all_wavs)
                    dataitem['config'] = corpus.encode(const.default_text_encoding) #[ord(c) for c in corpus]
                    dataitem['sourcefile'] = wavs[ int(permut[i]) ].encode(const.default_text_encoding)
                    #print (wavs[ int(i) ])

                    write_to_tfrecord(writers[i%const.num_tfrecord_shards], dataitem)
                    counts[i%const.num_tfrecord_shards] += 1
                if i % 10 == 0:
                    sys.stdout.write('\b\b\b\b\b\b\b\b%08i'%i)
                    sys.stdout.flush()


    for writer in writers:
        writer.close()
    print("... Done")


    for shard in range(const.num_tfrecord_shards):
        countfile = os.path.join(data_cache_dir, 'collection.%s.itemcount.%i' % ( corpus, shard ))
        np.savetxt(countfile, [counts[shard]], fmt='%i')

            
    return targetfiles, configs, counts





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
short_phone_to_class = {}



for k in phone_to_class.keys():
    if '#' not in k:
        kaldi_phone_to_short_phone[k] = re.sub(r'_[BEIS]', '',k)
        short_phone_to_kaldi_phone[re.sub(r'_[BEIS]', '',k)] = k

ct = 1
for k in short_phone_to_kaldi_phone.keys():
    if k == 'sil' or k == 'spn' or k == 'nsn' or k == '<eps>':
        short_phone_to_class[k] = 0
    else:
        short_phone_to_class[k] = ct
        ct += 1

class_to_class = {}
for k,p in class_to_phone.items():
    p = re.sub(r'_[BEIS]', '',p)
    if p in short_phone_to_class.keys():
        class_to_class[int(k)] = short_phone_to_class[re.sub(r'_[BEIS]', '',p)]
        


        
# Create alignment ctm files:

for dataset in ['train', 'devel', 'test']:
    for alignmentset in alignmentdirs[dataset]:
    
        alignmentdir = os.path.join(alignmentbasedir, alignmentset )


        utterance_record_dir = os.path.join(data_cache_dir, 'utterances', dataset)
        phoneme_record_dir =  os.path.join(data_cache_dir, 'phonemes', dataset)

        mkdir(utterance_record_dir)
        mkdir(phoneme_record_dir)
        
        promptsfile = os.path.join(labelbasedir, alignmentset.replace('tri3b_ali_', ''),  'text')
        filescpfile = os.path.join(labelbasedir, alignmentset.replace('tri3b_ali_', ''),  'wav.scp')
        starttimefile = os.path.join(labelbasedir, alignmentset.replace('tri3b_ali_', ''),  'feats.scp')

        #files_that_survived_cleaning_file = os.path.join(cleaned_labelbasedir, alignmentset.replace('tri3b_ali_', ''),  'feats.scp')

        #files_that_survived_cleaning = {}

        starttimescp={}

        print("Reading filepaths from %s" % starttimefile)
        for l in open(starttimefile,'r').readlines():
            f,p = l.strip().split(' ',1)
            #files_that_survived_cleaning[re.sub(r'\-[0-9]$',f)] = 1

            starttime = re.search(r'\[([0-9]+)\:[0-9]+\]', p)
            if starttime is None:
                #print("problem with starttime, continuing")
                #print(f,p)
                #print(starttime)
                #continue
                starttimescp[f] = 0
            else:
                starttimescp[f] = float(starttime.group(1))/100

        i = 1

        print("Checking for file %s/ali.%i.gz" % (alignmentdir, i))
        while os.path.exists("%s/ali.%i.gz" % (alignmentdir, i) ):
            outfile='/tmp/%s_ali_for_tfrecords.%i' %(alignmentset, i)
            if os.path.exists(outfile):
                print("File %s exists already" % outfile)
            else:
                print("Outputting alignments to %s" % outfile)
                call( [ali_to_phones,
                       '--ctm-output',
                       modelfile,
                       'ark:gunzip -c %s/ali.%i.gz |' % (alignmentdir, i),
                       outfile ])
            i+=1
            numfiles = i


        # First, a statistics run:

        phone_lengths = {}
        time_lengths = {}

        chopped_phone_lengths = {}
        chopped_time_lengths = {}

        corpora = []

        sourcefiles = {}
        starttimes = {}
        durations = {}
        phoneseqs = {}
        classseqs = {}
        wordseqs = {}
        ages = {}
        genders = {}
        endsilencelengths = {}

        phonemes = {}
        phonemes_for_files = {}

        for i in range(1, numfiles):
            print("Processing ali-file %i" % i)
            oldf = ''
            plen=0
            corpus=''
            ct = 0
            for l in open('/tmp/%s_ali_for_tfrecords.%i' %(alignmentset, i), 'r').readlines():
                f,_,s,d,p = l.strip().split()
                plen += 1
                if f != oldf:
                    if oldf is not '':
                        phone_lengths[corpus].append(int(plen))
                        time_lengths[corpus].append(float(olde))
                    plen=1
                    corpus=f[:20]
                    if corpus not in phone_lengths.keys():
                        phone_lengths[corpus] = []
                        time_lengths[corpus] = []
                oldf = f    
                olde = float(s)+float(d) # start + duration


            phone_lengths[corpus].append(int(plen))
            time_lengths[corpus].append(float(olde))


            oldf = ''
            plen=0
            corpus=''
            ct = 0

            cseq=[]
            pseq=[]

            begintime=0
            ind_ph=[]

            for l in open('/tmp/%s_ali_for_tfrecords.%i' %(alignmentset, i), 'r').readlines():

                f,_,s,d,p = l.strip().split()
                if debug:
                    print(f,s,d,class_to_phone[p])
                pseq.append( class_to_phone[p].split('_')[0] )
                cseq.append( int(p) )

                #if (class_to_phone[p] == 'sil' and float(d) > 0.4) or f != oldf:
                if f != oldf:
                    if oldf is not '':
                        #if 'spn' not in pseq and 'spn_S' not in pseq:
                            #if plen==3:
                            #    print (pseq)
                            #chopped_phone_lengths[corpus].append(int(plen))
                            ##chopped_time_lengths[corpus].append(float(olde))
                            #chopped_time_lengths[corpus].append(float(s)+0.75*float(d))
                            sourcefiles[corpus].append(oldf)
                            starttimes[corpus].append(begintime)
                            if debug:
                                print("Adding duration",float(olde), '+', 0.5*float(d), '-', begintime,'=',float(olde) + 0.5*float(d) - begintime)
                            if class_to_phone[p] == 'sil':
                                endsilencelengths[corpus].append(float(float(d)))
                                durations[corpus].append(float(olde) + float(oldd) - begintime)
                            else:
                                endsilencelengths[corpus].append( 0.0 )
                                durations[corpus].append(float(olde) + float(d) - begintime)
                            phoneseqs[corpus].append(pseq)
                            classseqs[corpus].append(cseq)

                            ages[corpus].append(age)
                            genders[corpus].append(gender)
                            phonemes_for_files[corpus].append(ind_ph)

                    ind_ph = [] # [float(s),float(d),int(p), class_to_phone[p]] ]

                    pseq = [class_to_phone[p]]
                    cseq = [int(p)]
                    plen=1

                    age=f[41:43]
                    if age == '--':
                        print("Age problem: %s %s"%(age, f))
                    gender=f[40]

                    #if class_to_phone[p] == 'sil':                
                    #    begintime = float(s) #+ 0.5 * float(d)
                    #else:
                    #    begintime = float(s)
                    begintime = 0

                    if debug:
                        print('begintime =',begintime)
                    corpus=f[:40]
                    if corpus not in corpora:
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
                oldf = f    
                olde = float(s)+float(d) # start + duration
                oldd = float(d)
                olds = float(s)

                #if 'spn' not in class_to_phone[p]:
                # Write phoneme to tfrecords:
                ind_ph.append([float(s),float(s)+float(d),int(p), class_to_phone[p]])
                plen += 1
                #if ct == 200:
                #    break
            #chopped_phone_lengths[corpus].append(int(plen))
            #chopped_time_lengths[corpus].append(float(olde))
            if plen > 2:
                sourcefiles[corpus].append(oldf)
                starttimes[corpus].append(begintime)
                phoneseqs[corpus].append(pseq)
                classseqs[corpus].append(cseq)
                ages[corpus].append(age)
                genders[corpus].append(gender)
                if class_to_phone[p] == 'sil':
                    endsilencelengths[corpus].append(float(d))
                    durations[corpus].append(float(olde) + float(d) - begintime)
                else:
                    endsilencelengths[corpus].append(0.0)
                    durations[corpus].append(float(olde) + float(d) - begintime)
                phonemes_for_files[corpus].append(ind_ph)
            #break

        #for i in np.argsort(np.array( durations )):
        #    print(sourcefiles[i],starttimes[i], durations[i], phoneseqs[i])

        #for c in phone_lengths.keys():
        #    print(c)
        #    #print(np.bincount(phone_lengths[c]))
        #    print(np.bincount(chopped_phone_lengths[c]))
        #    print(np.histogram(chopped_time_lengths[c],np.arange(0,4,0.25)))



        for k in durations.keys():
            durations[k]=np.array(durations[k])

        selected_corpora = ['pfstar_enuk____train_clean_wav__________',
                            'speechdat-fi___train_tele_FIA___________',
                            'speecon-fi_____train_clean_FI0__________',
                            'speecon-fi_____train_noisy_FI0__________',
                            'speeconkids-fi_train_clean_FI0__________',
                            'spraakbanken_setrain_random17k_wav______',
                            'tidigits_______children_train_wav_______',
                            'wsj1___________wsj1_si_tr_s_wv1_________',
                            'wsjcam0________si_tr_clean_wv1__________']

        selected_mics = ['wav', 'wv1', 'wav']

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


        for k in corpora:
          if 'FI2' not in k and 'FI3' not in k:
            #ct, _ = np.histogram(durations[k], [0.1, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0 , 4.0, 6.0, 8.0, 30.0])
            #recordcount = []
            #print(ct)
            for i in range(len(timebins)):

              timebin = timebins[i]
              recordfiles = []

              n = len(np.intersect1d(np.where(np.array(durations[k]) > timebin[0])[0], np.where(np.array(durations[k]) <= timebin[1])[0]))
              recordfilecount=int(np.ceil(n / utts_per_record ))

              if recordfilecount == 0:
                  continue

              utterance_writers = []
              phoneme_writers = []

              utterance_counts = []
              phoneme_counts = []

              utterance_counter_files = []
              phoneme_counter_files = []

              uttminct = np.inf
              phminct = np.inf
              for shard in range(recordfilecount):
                  uttctfile = os.path.join(utterance_record_dir, 'utterance_collection.%s.%0.1f-%0.1fs.counter.%i' % ( k, timebin[0], timebin[1], shard ))
                  print('checking %s' % uttctfile)
                  utterance_counter_files.append(uttctfile )

                  if os.path.exists( uttctfile ):
                      uttminct = min(uttminct, np.loadtxt(uttctfile).item())
                  else:
                      uttminct = 0

                  phctfile = os.path.join(phoneme_record_dir, 'phoneme_collection.%s.%0.1f-%0.1fs.counter.%i' % ( k, timebin[0], timebin[1], shard ))

                  print('checking %s' % phctfile)
                  phoneme_counter_files.append( phctfile )          


                  if os.path.exists( phctfile ):
                      phminct = min(phminct, np.loadtxt(phctfile).item())
                  else:
                      phminct = 0

                  print("utt min %i ph min %i" % (uttminct,phminct))            

              if uttminct > 0 and phminct > 0:
                  print('utterance_collection.%s.%0.2fs.tfrecord.%i' % ( k, timebin[1], shard ), 'already done')
                  print('phoneme_collection.%s.%0.2fs.tfrecord.%i' % ( k, timebin[1], shard ), 'already done')
                  continue

              print("%i candidates, put into following files:" % n)              
              for shard in range(recordfilecount): 
                  print("  Shard", shard, "of", recordfilecount)
                  utterance_targetfile = os.path.join(utterance_record_dir, 'utterance_collection.%s.%0.1f-%0.1fs.tfrecord.%i' % ( k, timebin[0], timebin[1], shard ))
                  utterance_writers.append (tf.python_io.TFRecordWriter(utterance_targetfile))
                  phoneme_targetfile = os.path.join(phoneme_record_dir, 'phoneme_collection.%s.%0.1f-%0.1fs.tfrecord.%i' % ( k, timebin[0], timebin[1], shard ))
                  phoneme_writers.append (tf.python_io.TFRecordWriter(phoneme_targetfile))
                  utterance_counts.append(0)
                  phoneme_counts.append(0)

                  np.savetxt(phoneme_counter_files[shard], [-1], fmt='%i')
                  np.savetxt(utterance_counter_files[shard], [-1], fmt='%i')

                  print("  ",utterance_targetfile)
                  print("  ",phoneme_targetfile)
              cnt = 0

              for j in np.random.permutation(np.intersect1d(np.where(np.array(durations[k]) > timebin[0])[0], np.where(np.array(durations[k]) <= timebin[1])[0])):
                f = sourcefiles[k][j]

                wseq = prompts[f]
                if '<UNK>' in wseq:
                    print ("<UNK> in prompt of %s" % f)
                    continue      

                lngo = re.search('..\_..\_', wseq)
                if lngo is None:
                    continue
                lng = lngo.group(0)[:-1]
                wseq = wseq.replace(lng+'_','')

                src =  sourcefiles[k][j]
                if src in filescp.keys():
                    filepath = filescp[ src ]
                elif src[:-2] in filescp.keys():
                    filepath = filescp[ src[:-2] ]
                else:
                    print("Could not find sourcefile %s" % src)
                    continue
                starttime = starttimes[k][j]
                duration = durations[k][j]
                endsil = endsilencelengths[k][j]

                startoffset = 0 #starttimescp[f]
                if startoffset != 0:
                    print("File %s start offset %f" % (f, startoffset))

                gender = genders[k][j]
                if gender.lower() != 'm' and gender.lower() != 'f':
                    print ("Gender '%s' not m or f for %s" % (gender, f))
                    continue

                try:
                    age = int(ages[k][j])
                except ValueError:
                    print ("Age '%s' not an integer for %s" % (ages[k][j],f))
                    continue

                #if j not in phonemes_for_files[k].keys():
                #    print("Trouble: No individual phonemes for index %i file %s" % (j,f))
                #    continue
                individual_phonemes = phonemes_for_files[k][j]

                pseq = phoneseqs[k][j]
                if ' '.join(pseq) == 'sil s p n sil':
                    continue

                cseq = classseqs[k][j]

                writer = np.argmin(utterance_counts)
                phoneme_counts = extract_and_save_to_record(utterance_writer = utterance_writers[writer],
                                                            phoneme_writers = phoneme_writers,
                                                            phoneme_writer_counters = phoneme_counts,
                                                            sourcefile = f,
                                                            filepath = filepath,
                                                            lng = lng,
                                                            wordseq = wseq,
                                                            phoneseq = pseq,
                                                            classseq = cseq,
                                                            starttime = starttime,
                                                            duration = duration,
                                                            individual_phonemes = individual_phonemes,
                                                            gender = gender,
                                                            age = age,
                                                            endsillength = endsil,
                                                            starttimeoffset = startoffset,
                                                            bucketlength=timebin[1]

                )
                #print(filescp[ [sourcefiles[k][j]] ],timebin[0],'<',"%0.2f" %durations[k][j],'<',timebin[1], phoneseqs[k][j])
                utterance_counts[writer] += 1
                cnt += 1
                #if cnt>10:
                #  break
              for shard in range(len(utterance_counter_files)):
                  np.savetxt(utterance_counter_files[shard], [utterance_counts[shard]], fmt='%i')
                  np.savetxt(phoneme_counter_files[shard], [phoneme_counts[shard]], fmt='%i')
              print("%i utterance segments processed" % cnt)

