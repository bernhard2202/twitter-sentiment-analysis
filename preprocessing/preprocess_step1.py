import re

FULL_POS_ORIG_FILE_NAME = "../data/train/train_pos_full_orig.txt"
FULL_NEG_ORIG_FILE_NAME = "../data/train/train_neg_full_orig.txt"
TEST_ORIG_FILE_NAME = "../data/test/test_data_orig.txt"

FULL_POS_FILE_NAME = "../data/train/train_pos_full.txt"
FULL_NEG_FILE_NAME = "../data/train/train_neg_full.txt"
TEST_FILE_NAME = "../data/test/test_data.txt"

def main():
    # patterns for handling numbers
    digit_char_digit = re.compile(r"\d+([a-z]+)\d+")      # substitute with <alphanum>
    char_digit = re.compile(r"[a-z]+\d+")               # substitute with <alphanum>
    num = re.compile(r"\d+")                            # substitute with <num>

    # specific patters for ending of line
    adv1_pattern = re.compile(r"\([^\(\)]*\.\.\.\s<url>$")
    adv2_pattern = re.compile(r"\([^\(\)]*\s<url>$")


    # for fin, fout in [("../data/train/debug_neg.txt", "../data/train/debug_step1.txt")]:
    for fin, fout in [(FULL_POS_ORIG_FILE_NAME, FULL_POS_FILE_NAME), (FULL_NEG_ORIG_FILE_NAME, FULL_NEG_FILE_NAME),
                      (TEST_ORIG_FILE_NAME, TEST_FILE_NAME)]:
        with open(fin, 'r') as f, open(fout, 'w') as out:
            print("start processing file:\t\t"+fin)
            line_cnt = 0
            for line in f:
                line_cnt += 1
                if line_cnt % 10000 == 0:
                    print(line_cnt)

                result = []   # here we accumulate the result line after the processing
                for word in line.split():
                    if word[0] == '#':
                        temp = re.sub(num, "<num>", word)
                        result.append(temp)
                        if temp != word:
                            pass
                            #print("{}\t\t-->\t\t{}".format(word, temp))
                        continue

                    # handle specific exceptions. like <3 we don't want it to be converted to <num>
                    # TODO(andrei): Extract this as a separate list of custom emoticons.
                    if word in ["<3"]:  # exception list
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
                        temp = re.sub(num, "<num>", word)
                        result.append(temp)
                        if temp != word:
                            pass
                            #print("{}\t\t-->\t\t{}".format(word, temp))


                # check some specific patterns in line. probably useless
                if bool(re.search(adv1_pattern, line)):
                    result.append("<adv1>")
                elif bool(re.search(adv2_pattern, line)) and not line.endswith("( <user> live on <url>\n") \
                        and not line.endswith("( <user> <url>\n"):
                    result.append("<adv2>")

                out.write(' '.join(result) + "\n")


if __name__ == "__main__":
    main()
