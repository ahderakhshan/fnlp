from hazm import stopwords_list

stopwords = stopwords_list()
stopwords.append("</s>")
stopwords.append("های")
for i in ["خوب", "بد", "ممنون", "عالی", "تشکر", "متوسط", "خیلی", "بسیار", "کم","متفاوت" , "واقعا", "شاید"]:
    if i in stopwords:
        stopwords.remove(i)


def remove_unnecessary_tokens(words):
    result = []
    for word in words:
        if len(word) == 1 or word in result:
            continue
        if " " not in word and word not in stopwords:
            result.append(word)
        elif " " in word:
            splited_words = word.split(" ")
            if splited_words[0] == splited_words[1] and splited_words[0] not in result:
                result.append(splited_words[0])
            elif len(splited_words[0]) == 1 or splited_words[0] in stopwords:
                if not (len(splited_words[1]) == 1 or splited_words[1] in stopwords):
                    if splited_words[1] not in result:
                        result.append(splited_words[1])
            elif len(splited_words[1]) == 1 or splited_words[1] in stopwords:
                if not (len(splited_words[0]) == 1 or splited_words[0] in stopwords):
                    if splited_words[0] not in result:
                        result.append(splited_words[0])
            else:
                result.append(word)
    return result


def manage_common(words, other_words):
    result = []
    for word in words:
        to_add = True
        for other_word in other_words:
            if " " not in word and " " not in other_word:
                if word == other_word:
                    to_add = False
            if " " in word and " " not in other_word:
                if word.split(" ")[0] == other_word or word.split(" ")[1] == other_word:
                    to_add = False
            if " " not in word and " " in other_word:
                if word == other_word.split(" ")[0] or word == other_word.split(" ")[1]:
                    to_add = False
            if " " in word and " " in other_word:
                word1, word2 = word.split(" ")
                other_word1, other_word2 = other_word.split(" ")
                if word1 == other_word1 or word1 == other_word2 or word2 == other_word1 or word2 == other_word2:
                    to_add = False
        if to_add:
            result.append(word)
    return result


label_words =  {
    'علم و تکنولوژی': ['فناوری فناوری', 'فناوری', 'فناوری سامسونگ', 'فناوری اپل', 'سامسونگ', 'فناوری تکنولوژی', 'فناوری افزار', 'اپل اپل', 'فناوری موبایل', 'فناوری ها', 'گوشی', 'فناوری ...', 'سامسونگ ...', 'فناوری ی', 'گوشی موبایل', 'خودرو', 'اخبار موبایل', 'فناوری .', 'طراحی', 'گوشی ...', 'اخبار ...'],
    'بازی ویدیویی': ['Xbox بازی', 'گیم', 'بازی', 'گیم پلی', 'تری بازی', 'بازی [...]', 'بازی ها', 'بازی آنلاین', 'اخبار بازی', 'بازی اخبار', 'بازی لر', 'گیم لر', 'بازی .', 'ویدیو', 'مایکروسافت', 'گرافیک'],
    'هنر و سینما': ['سینمایی', 'سینما', 'سینما فیلم', 'فیلم سینما', 'سینمای سینما', 'مستند', 'نقد فیلم', 'موسیقی', 'هنر', 'فیلم م', 'فیلم کار', 'فیلم ه'],
    'سلامت و زیبایی': ['سلامتی', 'تغذیه', 'پزشکی', 'سلامت پزشکی', 'سلامت', 'سلامت ی', 'سلامت ...', 'سلامت .', 'چرا ...', 'ورزشی', 'فوتبال ی', 'جام ملی', 'فوتبال ملی', 'فوتبال ها', 'جام ورزشی', 'فوتبال ...', 'جام فوتبال', 'فوتبال'],
    'کتاب و ادبیات': ['ادبیات', 'شعر', 'نویسنده', 'کتاب', 'کتاب خواندن', 'کتاب صوتی', 'خواندن', 'داستان', 'داستان داستان', 'کتاب ها', 'کتاب رمان', 'زندگی', 'زندگی زندگی', 'کتاب ی', 'داستان رمان', 'کتاب ...', 'رمان', 'کتاب .', 'اما ...'],
    'راهنمای خرید': ['خرید', 'خرید خرید', 'لوازم خرید', 'فروشگاه', 'راهنمای خرید', 'خرید ی', 'فروش', 'خرید ...', 'خرید .', 'خرید لباس', 'رنگ کالا', 'راهنمای .', 'راهنمای خودرو', 'راهنمای ی', 'راهنمای ...', 'رنگ ...', 'دکوراسیون', 'ماشین اصلاح', 'کفش', 'راهنمای لباس', 'آموزش', 'باغ بانی', 'راهنمای کودک', 'لباس'],
    'عمومی': ['عمومی', 'دانلود فیلم', 'بازی سول', 'بازی باکس']
}


for label, words in label_words.items():
    label_words[label] = remove_unnecessary_tokens(words)

final_label_words = {k: [] for k in label_words.keys()}

for label, words in label_words.items():
    other_words = []
    for other_label in label_words.keys():
        if other_label != label:
            other_words += label_words[other_label]
    new_words = manage_common(words, other_words)
    final_label_words[label] = new_words

print(final_label_words)


