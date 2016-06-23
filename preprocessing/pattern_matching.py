import re

"""
Searches for special patterns in the original tweets.
Those patterns are then handled individually, eg:
  * numbers are replaced with <num>
  * alphanumerics are replaced with <alphanum>

This has to run before the vocabulary is build!
"""

FULL_POS_ORIG_FILE_NAME = "../data/train/train_pos_full_orig.txt"
FULL_NEG_ORIG_FILE_NAME = "../data/train/train_neg_full_orig.txt"
TEST_ORIG_FILE_NAME = "../data/test/test_data_orig.txt"

FULL_POS_FILE_NAME = "../data/train/train_pos_full.txt"
FULL_NEG_FILE_NAME = "../data/train/train_neg_full.txt"
TEST_FILE_NAME = "../data/test/test_data.txt"

# TODO(Bernhard): look for a emoticon list on the web
# TODO(Bernhard): look weather those emoticons survive the nex preprocessing step
#                 or if we should replace them by tags like <positive_emoticon> and <neg...

#  handle specific exceptions. like <3 we don't want it to be converted to <num>
EMOTICONS = ["<3", ":3"]


def main():
    # patterns for handling numbers
    digit_char_digit = re.compile(r"\d+([a-z]+)\d+")      # substitute with <alphanum>
    char_digit = re.compile(r"[a-z]+\d+")               # substitute with <alphanum>
    num = re.compile(r"\d+")                            # substitute with <num>

    # specific patters for ending of line
    # Disabled for the moment, as they are a little to data-specific.
    # adv1_pattern = re.compile(r"\([^\(\)]*\.\.\.\s<url>$")
    # adv2_pattern = re.compile(r"\([^\(\)]*\s<url>$")

    for fin, fout in [(FULL_POS_ORIG_FILE_NAME, FULL_POS_FILE_NAME),
                      (FULL_NEG_ORIG_FILE_NAME, FULL_NEG_FILE_NAME),
                      (TEST_ORIG_FILE_NAME, TEST_FILE_NAME)]:
        with open(fin, 'r') as f, open(fout, 'w') as out:
            print("start processing file:\t\t"+fin)
            line_cnt = 0
            for line in f:
                line_cnt += 1
                if line_cnt % 10000 == 0:
                    print(line_cnt)

                # here we accumulate the result line after the processing
                result = []
                for word_index, word in enumerate(line.split()):
                    if word[0] == '#':
                        temp = re.sub(num, "<num>", word)
                        result.append(temp)
                        if temp != word:
                            pass
                            #print("{}\t\t-->\t\t{}".format(word, temp))
                        continue

                    #  handle specific exceptions. like <3 we don't want it to be converted to <num>
                    if word in EMOTICONS:  # exception list
                        result.append(word)
                        continue

                    # seach for pattern:   digits chars digits
                    match_obj = re.match(digit_char_digit, word)
                    if match_obj:       #if it matched
                        if match_obj.group(1) == "x":     # for digits x digits we have special treatment e.g. 1366x768 --> <num>x<num>
                            result.append("<num>x<num>")
                            #print("{}\t\t-->\t\t{}".format(word, "<num>x<num>"))
                        else:
                            result.append("<alphanum>")
                            #print("{}\t\t-->\t\t{}".format(word, "<alphanum>"))
                    elif bool(re.match(char_digit, word)):
                        result.append("<alphanum>")
                        #print("{}\t\t-->\t\t{}".format(word, "<alphanum>"))
                    else:
                        if fin == TEST_ORIG_FILE_NAME and word_index == 0:
                            # The test data file is indexed, unlike the positive
                            # and negative training ones. We don't want to
                            # replace the index with <num>!
                            searched_part = word[(word.index(',') + 1):]

                            # Since 're.sub' will only substitute in the
                            # 'searched_part', we want to make sure we don't
                            # forget to add the first part of the line to the
                            # output.
                            result_chunk = word[0:word.index(',') + 1]
                        else:
                            searched_part = word
                            result_chunk = ""

                        temp = re.sub(num, "<num>", searched_part)
                        result_chunk += temp
                        result.append(result_chunk)

                        # if temp != word:
                        #     print("{}\t\t-->\t\t{}".format(word, temp))


                # check some specific patterns in line. probably useless
                # Disabled for being too data-specific.
                # if bool(re.search(adv1_pattern, line)):
                #     result.append("<adv1>")
                # elif bool(re.search(adv2_pattern, line)) and not line.endswith("( <user> live on <url>\n") \
                #         and not line.endswith("( <user> <url>\n"):
                #     result.append("<adv2>")

                out.write(' '.join(result) + "\n")


if __name__ == "__main__":
    main()
