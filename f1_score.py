def longest_common_subsequence(str1, str2):
    m = len(str1)
    n = len(str2)

    # LCS를 저장하기 위한 2D 배열 초기화
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # LCS 계산
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 최장 공통 부분열의 길이
    lcs_length = dp[m][n]

    # 최장 공통 부분열 구하기
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    lcs.reverse()

    return lcs_length, ''.join(lcs)

"""
txt_ocr: ocr 예측 결과
txt_gt:  ground truth
lcs_length: LCS 문자열 길이
lcs_sequence: LCS에 해당하는 문자열
"""
def calc_f1_score(txt_gt, txt_ocr):
    # 아무 데이터 없는 경우 0을 출력
    if len(txt_gt) < 1 or len(txt_ocr) < 1:
        return 0
    lcs_length, lcs_sequence = longest_common_subsequence(txt_ocr, txt_gt)
    precision = lcs_length/len(txt_ocr)
    recall = lcs_length/len(txt_gt)

    if precision + recall == 0:
        f1_score = 0  # 예외 처리: 분모가 0인 경우
        return 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
