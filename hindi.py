#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %%capture
# !pip install datasets==1.4.1
# !pip install transformers==4.4.0
# !pip install torchaudio
# !pip install librosa
# !pip install jiwer


# In[2]:


# pip install datasets --upgrade


# In[3]:


# !wget https://drive.google.com/file/d/1vv37ceeqHoX1W7coXVpi1ms7v227eEp1/view?ts=6581246f&pli=1


# In[4]:


# !pip install gdown


# In[5]:


# !gdown --id 1vv37ceeqHoX1W7coXVpi1ms7v227eEp1


# In[6]:


# !unzip -P "password" "/content/HACKATHON_FILES.zip" -d "/content/drive/MyDrive/projects"


# In[7]:


import pandas as pd
# Example with orient parameter
hin = pd.read_json('/home/21bt04089/data_sih/HACKATHON_FILES/hindi.json', orient='records')  # 'records', 'split', 'index', 'columns', 'values'
hin.info()
hin.describe()
dict(hin)


# In[8]:


train=hin[:6600]
test=hin[6601:]


# In[9]:


test


# In[10]:


train


# In[ ]:





# In[ ]:





# In[11]:


from datasets import Dataset, load_metric

train= Dataset.from_pandas(train,split='train+validation')
print(train)
test = Dataset.from_pandas(test)


# In[12]:


train = train.remove_columns(['gender'])
test = test.remove_columns(['gender'])


# In[13]:


print(train)


# In[14]:


def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train.column_names)
vocab_test = test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=test.column_names)


# In[15]:


vocab_train


# In[16]:


vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
print(vocab_dict)


# In[17]:


print(len(vocab_dict))


# In[18]:


vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]


# In[19]:


vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(len(vocab_dict))


# In[20]:


import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)


# In[21]:


from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")


# In[22]:


from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)


# In[23]:


from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# In[24]:


processor.save_pretrained("/home/21bt04089/sihcode/model_hi/")


# In[25]:


train


# In[ ]:





# In[26]:


# # import torchaudio
# # import os


# # def speech_file_to_array_fn(batch):
# #     # Extract the directory path from the file path
# #     directory_path = "/home/21bt04089/data_sih/HACKATHON_FILES/HACKATHON/HINDI"

# #     # Extract the file name from the original path
# #     file_name = os.path.basename(batch["filepath"])

# #     # Construct the full path by joining the directory path and the file name
# #     full_path = os.path.join(directory_path, file_name)

# #     speech_array, sampling_rate = torchaudio.load(full_path)
# #     batch["speech"] = speech_array[0].numpy()
# #     batch["sampling_rate"] = sampling_rate
# #     batch["target_text"] = batch["text"]
# #     return batch
# import torchaudio
# import os
# import librosa
# import numpy as np
# from pydub import AudioSegment
# ​
# def apply_augmentation(audio, sampling_rate):
#     # Example: Time Stretching/Compression
#     speed_factor = np.random.uniform(0.7, 1.3)
#     audio = librosa.effects.time_stretch(audio, speed_factor)
# ​
#     # Example: Pitch Shifting
#     pitch_factor = np.random.uniform(-2, 2)  # pitch shift up or down by 2 semitones
#     audio = librosa.effects.pitch_shift(audio, sampling_rate, n_steps=pitch_factor)
# ​
#     # Example: Noise Injection
#     noise = np.random.normal(0, 0.005, len(audio))  # adjust the standard deviation as needed
#     audio = audio + noise
# ​
#     # Example: Amplitude Scaling
#     volume_factor = np.random.uniform(0.5, 1.5)
#     audio = audio * volume_factor
# ​
#     return audio
# ​
# def speech_file_to_array_fn(batch):
#     # Extract the directory path from the file path
#     directory_path = "/kaggle/working/hindi_data"
# ​
#     # Extract the file name from the original path
#     file_name = os.path.basename(batch["filepath"])
# ​
#     # Remove the ".mp3" extension if present and add the ".wav" extension
#     file_name = os.path.splitext(file_name)[0] + ".wav"
# ​
#     # Construct the full path by joining the directory path and the modified file name
#     full_path = os.path.join(directory_path, file_name)
    
#     try:
#         # Attempt to load the audio file
#         speech_array, sampling_rate = torchaudio.load(full_path)
# ​
#         # Apply augmentation
#         augmented_audio = apply_augmentation(speech_array[0].numpy(), sampling_rate)
# ​
#         batch["speech"] = augmented_audio.tolist()  # Convert to list
#         batch["sampling_rate"] = sampling_rate
#         batch["target_text"] = batch["text"]
#     except Exception as e:
#         print(f"Error loading audio file {full_path}: {str(e)}")
# ​
#     return batch
# ​



# # Assuming train and test are your datasets
# train = train.map(speech_file_to_array_fn, remove_columns=train.column_names)
# test = test.map(speech_file_to_array_fn, remove_columns=test.column_names)



# import torchaudio
# import os
# import librosa
# import numpy as np
# from pydub import AudioSegment

# def apply_augmentation(audio, sampling_rate):
#     # Example: Time Stretching/Compression
#     speed_factor = np.random.uniform(0.7, 1.3)
#     audio = librosa.effects.time_stretch(audio, speed_factor)

#     # Example: Pitch Shifting
#     pitch_factor = np.random.uniform(-2, 2)  # pitch shift up or down by 2 semitones
#     audio = librosa.effects.pitch_shift(audio, sampling_rate, n_steps=pitch_factor)

#     # Example: Noise Injection
#     noise = np.random.normal(0, 0.005, len(audio))  # adjust the standard deviation as needed
#     audio = audio + noise

#     # Example: Amplitude Scaling
#     volume_factor = np.random.uniform(0.5, 1.5)
#     audio = audio * volume_factor

#     return audio

# def speech_file_to_array_fn(batch):
#     # Extract the directory path from the file path
#     directory_path = "/kaggle/input/audios/HACKATHON_FILES/HACKATHON/URDU"

#     # Extract the file name from the original path
#     file_name = os.path.basename(batch["filepath"])

#     # Remove the ".mp3" extension if present and add the ".wav" extension
# #     file_name = os.path.splitext(file_name)[0] + ".wav"

#     # Construct the full path by joining the directory path and the modified file name
#     full_path = os.path.join(directory_path, file_name)
    
#     try:
#         # Attempt to load the audio file
#         speech_array, sampling_rate = torchaudio.load(full_path)

#         # Apply augmentation
#         augmented_audio = apply_augmentation(speech_array[0].numpy(), sampling_rate)

#         batch["speech"] = augmented_audio.tolist()  # Convert to list
#         batch["sampling_rate"] = sampling_rate
#         batch["target_text"] = batch["text"]
#     except Exception as e:
#         print(f"Error loading audio file {full_path}: {str(e)}")

#     return batch

# # Assuming train and test are your datasets
# train = train.map(speech_file_to_array_fn, remove_columns=train.column_names)
# test = test.map(speech_file_to_array_fn, remove_columns=test.column_names)

import torchaudio
import os
import librosa
import numpy as np
from pydub import AudioSegment

def apply_augmentation(audio, sampling_rate):
    # Example: Time Stretching/Compression
    speed_factor = np.random.uniform(0.7, 1.3)
    # Corrected: time_stretch takes only one argument
    audio = librosa.effects.time_stretch(y=audio, rate=speed_factor)

    # Example: Pitch Shifting
    pitch_factor = np.random.uniform(-2, 2)  # pitch shift up or down by 2 semitones
    audio = librosa.effects.pitch_shift(y=audio, sr=sampling_rate, n_steps=pitch_factor)

    # Example: Noise Injection
    noise = np.random.normal(0, 0.005, len(audio))  # adjust the standard deviation as needed
    audio = audio + noise

    # Example: Amplitude Scaling
    volume_factor = np.random.uniform(0.5, 1.5)
    audio = audio * volume_factor

    return audio

def speech_file_to_array_fn(batch):
    # Extract the directory path from the file path
    directory_path = "/home/21bt04089/data_sih/HACKATHON_FILES/HACKATHON/HINDI"

    # Extract the file name from the original path
    file_name = os.path.basename(batch["filepath"])

    # Remove the ".mp3" extension if present and add the ".wav" extension
#     file_name = os.path.splitext(file_name)[0] + ".wav"

    # Construct the full path by joining the directory path and the modified file name
    full_path = os.path.join(directory_path, file_name)
    
    try:
        # Attempt to load the audio file
        speech_array, sampling_rate = torchaudio.load(full_path)

        # Apply augmentation
        augmented_audio = apply_augmentation(speech_array[0].numpy(), sampling_rate)

        batch["speech"] = augmented_audio.tolist()  # Convert to list
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["text"]
    except Exception as e:
        print(f"Error loading audio file {full_path}: {str(e)}")

    return batch

# Assuming train and test are your datasets
train = train.map(speech_file_to_array_fn, remove_columns=train.column_names)
test = test.map(speech_file_to_array_fn, remove_columns=test.column_names)




# In[27]:


train[20]


# In[28]:


train


# In[29]:


# from pydub import AudioSegment
# import os
# def convert_mp3_to_wav(mp3_file, output_folder):
#     sound = AudioSegment.from_mp3(mp3_file)
#     output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(mp3_file))[0] + ".wav")
#     sound.export(output_file, format="wav")
#     print(f"Converted MP3: {mp3_file} to {output_file}")

# def convert_flac_to_wav(flac_file, output_folder):
#     sound = AudioSegment.from_file(flac_file, format="flac")
#     output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(flac_file))[0] + ".wav")
#     sound.export(output_file, format="wav")
#     print(f"Converted FLAC: {flac_file} to {output_file}")
# def convert_all_to_wav(input_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     for filename in os.listdir(input_folder):
#         input_path = os.path.join(input_folder, filename)

#         if filename.lower().endswith(".mp3"):
#             convert_mp3_to_wav(input_path, output_folder)
#         elif filename.lower().endswith(".flac"):
#             convert_flac_to_wav(input_path, output_folder)
# input_folder = "/home/21bt04089/data_sih/HACKATHON_FILES/HACKATHON/HINDI"
# output_folder = "/home/21bt04089/data_sih/HACKATHON_FILES/hindi-wav"

# convert_all_to_wav(input_folder, output_folder)


# In[30]:


import librosa
import numpy as np

def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), batch['sampling_rate'], 16_000)
    batch["sampling_rate"] = 16_000
    return batch

train = train.map(resample, num_proc=4)
test = test.map(resample, num_proc=4)


# In[87]:


train


# In[31]:


import IPython.display as ipd
import numpy as np
import random

rand_int = random.randint(0, len(train))
print("Target text:", train[rand_int]["target_text"])
print("Input array shape:", np.asarray(train[rand_int]["speech"]).shape)
print("Sampling rate:", train[rand_int]["sampling_rate"])
ipd.Audio(data=np.asarray(train[rand_int]["speech"]), autoplay=True, rate=16000)



# In[32]:


rand_int = random.randint(0, len(train))

print("Target text:", train[rand_int]["target_text"])
print("Input array shape:", np.asarray(train[rand_int]["speech"]).shape)
print("Sampling rate:", train[rand_int]["sampling_rate"])


# In[33]:


def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

train_data = train.map(prepare_dataset, remove_columns=train.column_names, batch_size=8, num_proc=4, batched=True)
test_data = test.map(prepare_dataset, remove_columns=test.column_names, batch_size=8, num_proc=4, batched=True)


# In[88]:


# import torch
# torch.cuda.set_device(0)        
# import random
# from typing import List, Dict, Union, Optional
# from transformers import Wav2Vec2Processor
# from torch.nn.utils.rnn import pad_sequence
# from dataclasses import dataclass, field
# from typing import Any, Dict, List, Optional, Union

# @dataclass
# class DataCollatorCTCWithPadding:
#     def __init__(
#         self,
#         processor: Wav2Vec2Processor,
#         padding: Union[bool, str] = True,
#         max_length: Optional[int] = 10,
#         max_length_labels: Optional[int] = None,
#         pad_to_multiple_of: Optional[int] = None,
#         pad_to_multiple_of_labels: Optional[int] = None,
#         overlap_factor: Optional[float] = 0.5,
#     ):
#         self.processor = processor
#         self.padding = padding
#         self.max_length = max_length
#         self.max_length_labels = max_length_labels
#         self.pad_to_multiple_of = pad_to_multiple_of
#         self.pad_to_multiple_of_labels = pad_to_multiple_of_labels
#         self.overlap_factor = overlap_factor

#     def _call_(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         # split inputs and labels since they have to be of different lengths and need different padding methods
#         input_features = [{"input_values": feature["input_values"]} for feature in features]
#         label_features = [{"input_ids": feature["labels"]} for feature in features]

#         # Calculate the overlap length
#         overlap_length = int(self.max_length * self.overlap_factor)

#         # Create overlapping segments
#         segments = []
#         for feature in features:
#             audio_length = len(feature["input_values"][0])
#             start_positions = list(range(0, audio_length, overlap_length))
#             for start in start_positions:
#                 end = min(start + self.max_length, audio_length)
#                 segment = {"input_values": feature["input_values"][:, start:end], "labels": feature["labels"]}
#                 segments.append(segment)

#         # Pad input sequences to max_length
#         batch = self.processor.pad(
#             [{"input_values": item["input_values"]} for item in segments],
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         )
#         with self.processor.as_target_processor():
#             labels_batch = self.processor.pad(
#                 [{"input_ids": item["labels"]} for item in segments],
#                 padding=self.padding,
#                 max_length=self.max_length_labels,
#                 pad_to_multiple_of=self.pad_to_multiple_of_labels,
#                 return_tensors="pt",
#             )

#         # replace padding with -100 to ignore loss correctly
#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

#         batch["labels"] = labels

#         return batch
# # import torch
# # from transformers import Wav2Vec2Processor
# # from dataclasses import dataclass, field
# # from typing import Any, Dict, List, Optional, Union

# # @dataclass
# # class DataCollatorCTCWithPadding:
# #     processor: Wav2Vec2Processor
# #     padding: Union[bool, str] = True
# #     max_length: Optional[int] = None
# #     max_length_labels: Optional[int] = None
# #     pad_to_multiple_of: Optional[int] = None
# #     pad_to_multiple_of_labels: Optional[int] = None

# #     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
# #         # split inputs and labels since they have to be of different lengths and need different padding methods
# #         input_features = [{"input_values": feature["input_values"]} for feature in features]
# #         label_features = [{"input_ids": feature["labels"]} for feature in features]

# #         # Subsample short clips during batch creation
# #         batch = self.create_mixed_length_batch(input_features)

# #         with self.processor.as_target_processor():
# #             # Apply adaptive padding for labels
# #             labels_batch = self.processor.pad(
# #                 label_features,
# #                 padding=self.padding,
# #                 max_length=self.max_length_labels,
# #                 pad_to_multiple_of=self.pad_to_multiple_of_labels,
# #                 return_tensors="pt",
# #             )

# #         # replace padding with -100 to ignore loss correctly
# #         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

# #         batch["labels"] = labels

# #         return batch

#     # def create_mixed_length_batch(self, batch):
#     #     # Subsample short clips during batch creation
#     #     batch = random.sample(batch, len(batch) // 2)  # Sample short sequences
#     #     batch += random.choices([seq for seq in batch if len(seq["input_values"][0]) > self.max_length], k=len(batch) // 2)  # Sample long sequences
#     #     random.shuffle(batch)
#     #     return batch


# In[90]:


# import torch
# from transformers import Wav2Vec2Processor
# from torch.nn.utils.rnn import pad_sequence
# from dataclasses import dataclass, field
# from typing import List, Dict, Union, Optional

# @dataclass
# class DataCollatorCTCWithPadding:
#     def __init__(
#         self,
#         processor: Wav2Vec2Processor,
#         padding: Union[bool, str] = True,
#         max_length: Optional[int] = 10,
#         max_length_labels: Optional[int] = None,
#         pad_to_multiple_of: Optional[int] = None,
#         pad_to_multiple_of_labels: Optional[int] = None,
#         overlap_factor: Optional[float] = 0.5,
#     ):
#         self.processor = processor
#         self.padding = padding
#         self.max_length = max_length
#         self.max_length_labels = max_length_labels
#         self.pad_to_multiple_of = pad_to_multiple_of
#         self.pad_to_multiple_of_labels = pad_to_multiple_of_labels
#         self.overlap_factor = overlap_factor

#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         # Ensure your features list has the correct structure
#         # Example structure: [{"input_values": your_input_values_list, "labels": your_labels_list}, ...]

#         # Calculate the overlap length
#         overlap_length = int(self.max_length * self.overlap_factor)

#         # Create overlapping segments
#         segments = []

#         for feature in features:
#             if isinstance(feature["input_values"][0], float):
#                 # Handle the case where input_values[0] is a float
#                 # You might want to print or log some information to debug
#                 print("Unexpected type for input_values[0]: float")
#                 continue

#             # Check if it's a tensor, and convert to list if needed
#             if isinstance(feature["input_values"][0], torch.Tensor):
#                 feature["input_values"] = feature["input_values"][0].tolist()

#             # Ensure that the first element of input_values is a list
#             if not isinstance(feature["input_values"][0], (list, torch.Tensor)):
#                 # Handle the case where input_values[0] is not a list or tensor
#                 # You might want to print or log some information to debug
#                 print("Unexpected type for input_values[0]:", type(feature["input_values"][0]))
#                 continue

#             audio_length = len(feature["input_values"])
#             start_positions = list(range(0, audio_length, overlap_length))
#             for start in start_positions:
#                 end = min(start + self.max_length, audio_length)
#                 segment = {"input_values": feature["input_values"][start:end], "labels": feature["labels"]}
#                 segments.append(segment)

#         # Pad input sequences to max_length
#         batch = self.processor.pad(
#             [{"input_values": item["input_values"]} for item in segments],
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         )
#         with self.processor.as_target_processor():
#             labels_batch = self.processor.pad(
#                 [{"input_ids": item["labels"]} for item in segments],
#                 padding=self.padding,
#                 max_length=self.max_length_labels,
#                 pad_to_multiple_of=self.pad_to_multiple_of_labels,
#                 return_tensors="pt",
#             )

#         # replace padding with -100 to ignore loss correctly
#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

#         batch["labels"] = labels

#         return batch




# torch.cuda.set_device(0)

import torch
import os

# # Set the environment variable
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=256"

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

# torch.cuda.empty_cache()


# In[96]:


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


# In[97]:


wer_metric = load_metric("wer")


# In[98]:


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    print(label_str)
    print(pred_str)
    return {"wer": wer}


# In[99]:


from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True
)

# model=model.to('cuda:0')


# In[100]:


model.freeze_feature_extractor()


# In[104]:


from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="/home/21bt04089/sihcode/notebooks/drashti-hindi",
  group_by_length=True,
  per_device_train_batch_size=8,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=20,
  # fp16=True,
  save_steps=200,
  eval_steps=200,
  logging_steps=200,
  learning_rate=5e-4,#0.1
  warmup_steps=300,
  save_total_limit=2,
  use_cpu=False,
  # per_gpu_train_batch_size=8
)


# In[105]:


from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=processor.feature_extractor,
)
trainer.args._n_gpu = 0   


# In[1]:


trainer.train()    


# In[108]:


import gc
gc.collect()


# In[ ]:




