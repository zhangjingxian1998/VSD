# BLEU-N
用于机器翻译任务评价，常见有BLEU-1，BLEU-2，BLEU-3，BLEU-4。其中的数字表示连续单词的个数。BLEU-1衡量的是单词级别的准确性，高阶BLEU衡量的句子流畅性。

通常用来衡量一组机器产生的翻译句子集合(candidates)与一组人工翻译句子(references)的相似程度。
```
示例
candidate: The cat sat on the mat.
reference: The cat is on the mat.
```
### BLUE-1
candidate: **{The}** **{cat}** {sat} **{on}** **{the}** **{mat}**

reference: **{The}** **{cat}** {is} **{on}** **{the}** **{mat}**

$BLUE_1 = \frac{5}{6}^1$

### BLUE-2
candidate: **{The cat}** {cat sat} {sat on} **{on the}** **{the mat}**

reference: **{The cat}** {cat is} {is on} **{on the}** **{the mat}**

$BLUE_2 = BLUE_1 * \frac{3}{5}^\frac{1}{2}$

### BLUE-3
candidate: {The cat sat} {cat sat on} {sat on the} **{on the mat}**

reference: {The cat is} {cat is on} {is on the} **{on the mat}**

$BLUE_3 = BLUE_2 * \frac{1}{4}^\frac{1}{3}$

### BLUE-4
candidate: {The cat sat on} {cat sat on the} {sat on the mat}

reference: {The cat is on} {cat is on the} {is on the mat}

$BLUE_4 = BLUE_3 * \frac{0}{3}^\frac{1}{4}$
```
ratio = candidate_len / reference_len
if ratio<1:
    BLUE_n = BLUE_n * math.exp(1 - 1/ratio)
```
# METEOR
不知道
# ROUGE
主要考虑最大单词匹配长度

$score=\frac{(1+beta^2)*prec_{max}*rec_{max}}{rec_{max}+beta^2*prec_{max}}$
示例

candidate: **The cat** sat **on the mat**.

reference: **The cat** is **on the mat**.
```
prec_max = 3/len_ca
rec_max = 3/len_re
beta=1.2
```

# CIDEr

# SPICE

# Acc