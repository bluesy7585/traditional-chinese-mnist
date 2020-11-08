import os

def dataset_char_set(word_dir):
    dataset_char_dict = {}
    dir_num = 0
    for item in os.listdir(word_dir):
        fullpath = os.path.join(word_dir, item)
        if os.path.isdir(fullpath):
            dir_num += 1
            img_num = 0
            for img in os.listdir(fullpath):
                if img.endswith('png'):
                    img_num += 1
            dataset_char_dict[item] = dataset_char_dict.get(item, 0) + img_num

    print('{} char class in dataset'.format(dir_num))
    return dataset_char_dict


def get_available_char_set(names, img_dir):
    name_char_dict = {}
    for name in names:
        for ch_char in name:
            name_char_dict[ch_char] = name_char_dict.get(ch_char, 0) + 1

    total_ch_char_dict = dataset_char_set(img_dir)

    #print(len(total_ch_char_dict))
    available_set = []
    not_in_total_dict =[]
    for ch in name_char_dict:
        if ch in total_ch_char_dict:
            available_set.append(ch)
            # if total_ch_char_dict[ch] < 40:
            #    print(f'{ch} only has {total_ch_char_dict[ch]} samples.')
        else:
            not_in_total_dict.append(ch)

    print('{} char class in random names'.format(len(name_char_dict)))
    print('{} char class not in dataset'.format(len(not_in_total_dict)))
    print('{} available char class'.format(len(available_set)))
    return available_set


if __name__ == '__main__':

    filename='ch_names.txt'
    img_dir = 'cleaned_data'

    with open(filename,'r',encoding='utf-8') as f:
        names = f.readline().split(',')
    print('{} names'.format(len(names)))

    ava_set = get_available_char_set(names, img_dir)
