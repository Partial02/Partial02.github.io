import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 하이퍼파라미터 설정
args = {
    "train_path" : "/content/drive/MyDrive/em_train.txt",      # train 데이터 경로
    "test_path" : "/content/drive/MyDrive/em_test.txt",       # test 데이터 경로
    "class_num" : 2  # 이진 분류
}

# 데이터 로드
train = pd.read_table(args["train_path"], sep='\\s+', header=None)
test = pd.read_table(args["test_path"], sep='\\s+', header=None)

# 데이터 정렬
train_sorted = train.sort_values(by=13, ascending=True)

# 이진 클래스의 구간을 기록
train_zero2one = (train_sorted[13]==1).idxmax()
train_sorted = np.array(train_sorted)
test = np.array(test)
test_label = test[:, -1]

# 파라미터 기록용 및 초기화용 리스트 
record_pi, record_mu, record_sig, record_err = [], [], [], []
pi, mu, sig = None, None, None



# 파라미터 초기화 함수
def param(num_cluster, zero2one, N):
    # num_cluster: kfold의 개수. k라고 표기
    # zero2one: 학습데이터의 0과 1의 기준을 표기
    # N: 직전 셀의 train_len에 해당. 데이터셋 행의 개수

    # num_feature: 데이터의 feature 수. 13개
    num_feature = train_sorted.shape[1]

    # 파라미터 초기화
    ## pi는 pi_k를 저장하는 리스트. 이진클래스에 맞게 기록한다
    ## pi_k는 prior를 indentically distributed라고 가정하여 1/K로 계산
    pi = [[1.0 / int(args["class_num"]), 1.0 / int(args["class_num"])]]

    ## mu는 mu_k를 저장하는 리스트. 이진클래스에 맞게 기록한다
    ## mu_k는 random한 평균벡터로 설정할 수 있다
    ## 0에서 1 사이의 랜덤한 실숫값을 feature의 수에 맞게 벡터로 사용
    mu = [[np.random.rand(num_feature), np.random.rand(num_feature)]]

    ## sig는 sigma_k를 저장하는 리스트. 이진클래스에 맞게 기록한다
    ## triu를 활용하여 대칭행렬 형태로 만든다
    # arr = np.random.rand(num_feature**2)+2.1
    # temp_mat = np.triu(arr.reshape(num_feature, num_feature))
    # cov_mat = temp_mat.dot(temp_mat.T)
    # sig = [[cov_mat, cov_mat]]
    sig = [[np.identity(num_feature, float), np.identity(num_feature, float)]]

    ## kfold를 통해 train/validation을 나눌 토막의 길이 계산
    ## k_interval은 클래스 0, 1때의 각 구간 길이를 저장하는 array
    ## 몫 계산을 통해 에러를 막고, 가장 마지막 iter때 남은 모든 데이터는 한번에 처리한다
    k_interval = np.array([zero2one // num_cluster, (N - zero2one) // num_cluster])

    return pi, mu, sig, k_interval


# K-Fold 학습 함수
# 마지막에 이 함수의 k를 2부터 늘려가며 에러를 확인한다
def kfold(data, label, k, zero2one, N):
    # data: 학습데이터. k등분하여 하나는 validation에, 나머지들은 학습에 쓰인다.
    # label: 학습데이터의 레이블
    # k: kfold의 핵심 파라미터. 반복문으로 k를 2부터 늘려가며 K-Fold를 진행한다.
    # zero2one: 학습데이터의 0과 1의 기준을 표기

    # num_feature: 학습 데이터의 피처의 개수. 13개
    num_feature = data.shape[1]

    # pi, mu, sig: 3개의 파라미터. param()에게서 받아온다
    # k_interval: 0 클래스 / 1 클래스의 길이를 저장하는 array. param()에게서 받아온다
    pi, mu, sig, k_interval = param(k, zero2one, N)
    sum_error = 0

    ## 파라미터를 선언(갱신 전. 값은 아래에서 업데이트하므로 형태만 갖춘다)
    pi_k0 = 1
    pi_k1 = 1
    mu_k0 = np.zeros((1,num_feature))
    mu_k1 = np.zeros((1,num_feature))
    sig_k0 = np.identity(num_feature, float)
    sig_k1 = np.identity(num_feature, float)

    ## 파라미터를 선언(갱신 후. 값은 아래에서 업데이트하므로 형태만 갖춘다)
    pi_k0_new = 1
    pi_k1_new = 1
    mu_k0_new = np.zeros((1,num_feature))
    mu_k1_new = np.zeros((1,num_feature))
    sig_k0_new = np.identity(num_feature, float)
    sig_k1_new = np.identity(num_feature, float)

    for i in range(k):

        ## 검증 데이터의 구간 양끝을 클래스별로 지정
        intL0, intR0 = k_interval[0] * i, k_interval[0] * (i + 1)
        intL1, intR1 = zero2one + k_interval[1] * i, zero2one + k_interval[1] * (i + 1)

        ## 가장 마지막 구간에선 남은 데이터들을 모두 포함시켜 validation으로 편성한다
        if (i == k-1):
            intR0 = zero2one
            data_val0 = data[intL0:intR0]
            label_val0 = label[intL0:intR0]
            data_x0 = data[0:intL0]

            intR1 = len(data)
            data_val1 = data[intL1:intR1]
            label_val1 = label[intL1:intR1]
            data_x1 = data[zero2one:intL1]

        else:
            data_val0 = data[intL0:intR0]
            label_val0 = label[intL0:intR0]
            data_x0 = np.concatenate([data[0:intL0], data[intR0:zero2one]], axis=0)

            data_val1 = data[intL1:intR1]
            label_val1 = label[intL1:intR1]
            data_x1 = np.concatenate([data[zero2one:intL1], data[intR1:]], axis=0)

        ## 수렴할 때까지 학습 데이터를 학습시킨다
        iter_count = 0
        while True:
            
            ## (k-1/k)의 train 데이터 뭉치에 대한 각각의 w_k^t를 계산
            wkt0 = []
            wkt1 = []

            ## 최초 선언이라면 초깃값을 불러오고, 아니라면
            if (iter_count == 0):
                pi_k0, pi_k1 = pi[i][0], pi[i][1]
                mu_k0, mu_k1 = mu[i][0], mu[i][1]
                sig_k0, sig_k1 = sig[i][0], sig[i][1]
            else: # 가장 최근 i-iter에서 계산된 값을 되먹임
                pi_k0, pi_k1 = pi_k0_new, pi_k1_new
                mu_k0, mu_k1 = mu_k0_new, mu_k1_new
                sig_k0, sig_k1 = sig_k0_new, sig_k1_new

            ## 가중치(확률) w_k^t 계산
            ### 행렬식 부분 계산
            det_sigma_k0 = np.linalg.det(sig_k0)
            det_sigma_k1 = np.linalg.det(sig_k1)
            det_sqrt_inv0 = (1 / np.sqrt(det_sigma_k0))
            det_sqrt_inv1 = (1 / np.sqrt(det_sigma_k1))

            for m in range(len(data_x0)):
                ### 클래스 0에서의 지수 부분 계산
                diff0 = data_x0[m] - mu_k0
                exp_k0 = np.exp(-0.5 * np.dot(np.dot(diff0.T, np.linalg.inv(sig_k0)), diff0))
                
                ### 클래스 0에서의 분자 부분 계산
                numerator0 = det_sqrt_inv0 * exp_k0 * pi_k0
                wkt0.append(numerator0)

            
            for n in range(len(data_x1)):
                ### 클래스 1에서의 지수 부분 계산
                diff1 = data_x1[n] - mu_k1
                exp_k1 = np.exp(-0.5 * np.dot(np.dot(diff1.T, np.linalg.inv(sig_k1)), diff1))
                
                ### 클래스 1에서의 분자 부분 계산
                numerator1 = det_sqrt_inv1 * exp_k1 * pi_k1
                wkt1.append(numerator1)

            ### 분자를 분모로 나누어 최종 wkt 계산
            wkt0 = np.array(wkt0)
            wkt1 = np.array(wkt1)
            wkt0 = wkt0 / (wkt0.sum())
            wkt1 = wkt1 / (wkt1.sum())


            ## pi0, pi1 신규 계산
            pi_k0_new = np.array(wkt0).sum() / len(wkt0)
            pi_k1_new = np.array(wkt1).sum() / len(wkt1)

            ## mu0, sig0 신규 계산
            mu_k0_new = np.dot(wkt0, data_x0)
            for m in range(len(data_x0)):
                sig_k0_new += wkt0[m] * np.dot(data_x0[m].T, data_x0[m])
            sig_k0_new = sig_k0_new / wkt0.sum() - np.dot(mu_k0_new.T, mu_k0_new)

            ## mu1, sig1 신규 계산
            mu_k1_new = np.dot(wkt1, data_x1)
            for n in range(len(data_x1)):
                sig_k1_new += wkt1[n] * np.dot(data_x1[n].T, data_x1[n])
            sig_k1_new = sig_k1_new / wkt1.sum() - np.dot(mu_k1_new.T, mu_k1_new)


            ## 수렴을 판단하기 위해 이전(formal)과 이후(latter)의 Q함숫값 계산
            ExpQ_formal = wkt0[0]*(-0.5 * np.log(np.linalg.det(sig_k0)) -0.5 * np.dot(np.dot((data[0] - mu_k0).T, np.linalg.inv(sig_k0)), (data[0] - mu_k0))) + np.log(pi_k0)
            ExpQ_formal += wkt1[0]*(-0.5 * np.log(np.linalg.det(sig_k1)) -0.5 * np.dot(np.dot((data[0] - mu_k1).T, np.linalg.inv(sig_k1)), (data[0] - mu_k1))) + np.log(pi_k1)

            ExpQ_latter = wkt0[0]*(-0.5 * np.log(np.linalg.det(sig_k0_new)) -0.5 * np.dot(np.dot((data[0] - mu_k0_new).T, np.linalg.inv(sig_k0_new)), (data[0] - mu_k0_new))) + np.log(pi_k0_new)
            ExpQ_latter += wkt1[0]*(-0.5 * np.log(np.linalg.det(sig_k1_new)) -0.5 * np.dot(np.dot((data[0] - mu_k1_new).T, np.linalg.inv(sig_k1_new)), (data[1] - mu_k1_new))) + np.log(pi_k1_new)
            
            ## 만약 두 Q함숫값(log likelihood의 expectation)의 차가 1/1000 미만이라면
            iter_count += 1
            if (abs(ExpQ_formal - ExpQ_latter) < 0.001):
                ## 수렴으로 판단하고 새로운 값들을 리스트에 기록
                pi.append([pi_k0_new, pi_k1_new])
                mu.append([mu_k0_new, mu_k1_new])
                sig.append([sig_k0_new, sig_k1_new])
                break
            else:
                ## 아니라면 기존 wkt를 초기화하여 새로운 계산을 준비
                wkt0 = []
                wkt1 = []
                #sig_k0_new = np.identity(13, float)
                #sig_k1_new = np.identity(13, float)


        ## 검증 데이터로 에러 계산, 이후 파라미터 저장
        error = cal_error(data_val0, data_val1, label_val0, label_val1, pi[-1][0], pi[-1][1], mu[-1][0], mu[-1][0], sig[-1][0], sig[-1][0])
        sum_error += error

    return pi[-1][0], pi[-1][1], mu[-1][0], mu[-1][0], sig[-1][0], sig[-1][0], sum_error / k


# 검증 데이터로 에러를 계산하는 함수
def cal_error(v0, v1, l0, l1, p0, p1, m0, m1, s0, s1):
    val = np.vstack((v0, v1))
    label = np.concatenate((l0, l1))
    wkt0 = []
    wkt1 = []

    ## kfold 함수와 비슷한 구성
    det_sigma_k0 = np.linalg.det(s0)
    det_sigma_k1 = np.linalg.det(s1)
    det_sqrt_inv0 = (1 / np.sqrt(det_sigma_k0))
    det_sqrt_inv1 = (1 / np.sqrt(det_sigma_k1))

    ## kfold와 달리 클래스 0과 1에서 학습시키는 데이터가 val로 같다. 하나의 반복문으로 묶어 계산
    for m in range(len(val)):
        diff0 = val[m] - m0
        exp_k0 = np.exp(-0.5 * np.dot(np.dot(diff0.T, np.linalg.inv(s0)), diff0))
        numerator0 = det_sqrt_inv0 * exp_k0 * p0
        wkt0.append(numerator0)

        diff1 = val[m] - m1
        exp_k1 = np.exp(-0.5 * np.dot(np.dot(diff1.T, np.linalg.inv(s1)), diff1))
        numerator1 = det_sqrt_inv1 * exp_k1 * p1
        wkt1.append(numerator1)

    ## 계산된 값들을 토대로 wkt를 계산
    wkt0 = np.array(wkt0)
    wkt1 = np.array(wkt1)
    wkt0 = wkt0 / (wkt0.sum())
    wkt1 = wkt1 / (wkt1.sum())

    ## 이진 클래스 중 어디에 속할지 Expectation 계산
    ## acc는 정확도
    acc = 0

    for m in range(len(val)):
        class0 = wkt0[m]*(-0.5 * np.log(np.linalg.det(s0)) -0.5 * np.dot(np.dot((val[m] - m0).T, np.linalg.inv(s0)), (val[m] - m0))) + np.log(p0)
        class1 = wkt1[m]*(-0.5 * np.log(np.linalg.det(s1)) -0.5 * np.dot(np.dot((val[m] - m1).T, np.linalg.inv(s1)), (val[m] - m1))) + np.log(p1)
        ### 클래스의 기댓값이 더 큰 쪽으로 분류 후, 정확도를 체크한다
        if ((class0 > class1 and label[m] == 0) or (class0 < class1 and label[m] == 1)):
            acc += 1
    
    return 1 - (acc / len(val))

# 테스트 데이터에 대한 에러율 계산
def cal_test_err(data, label, pi, mu, sig):
    wkt0 = []
    wkt1 = []
    p0, p1 = pi[0], pi[1]
    m0, m1 = mu[0], mu[1]
    s0, s1 = sig[0], sig[1]

    det_sigma_k0 = np.linalg.det(s0)
    det_sigma_k1 = np.linalg.det(s1)
    det_sqrt_inv0 = (1 / np.sqrt(det_sigma_k0))
    det_sqrt_inv1 = (1 / np.sqrt(det_sigma_k1))

    for m in range(len(data)):
        diff0 = data[m] - m0
        exp_k0 = np.exp(-0.5 * np.dot(np.dot(diff0.T, np.linalg.inv(s0)), diff0))
        numerator0 = det_sqrt_inv0 * exp_k0 * p0
        wkt0.append(numerator0)

        diff1 = data[m] - m1
        exp_k1 = np.exp(-0.5 * np.dot(np.dot(diff1.T, np.linalg.inv(s1)), diff1))
        numerator1 = det_sqrt_inv1 * exp_k1 * p1
        wkt1.append(numerator1)

    wkt0 = np.array(wkt0)
    wkt1 = np.array(wkt1)
    wkt0 = wkt0 / (wkt0.sum())
    wkt1 = wkt1 / (wkt1.sum())

    ## 이진 클래스 중 어디에 속할지 Expectation 계산
    ## acc는 정확도
    acc = 0

    for m in range(len(data)):
        class0 = wkt0[m]*(-0.5 * np.log(np.linalg.det(s0)) -0.5 * np.dot(np.dot((data[m] - m0).T, np.linalg.inv(s0)), (data[m] - m0))) + np.log(p0)
        class1 = wkt1[m]*(-0.5 * np.log(np.linalg.det(s1)) -0.5 * np.dot(np.dot((data[m] - m1).T, np.linalg.inv(s1)), (data[m] - m1))) + np.log(p1)
        if ((class0 > class1 and label[m] == 0) or (class0 < class1 and label[m] == 1)):
            acc += 1

    return 1 - (acc / len(data))



# 전체 코드의 가장 메인이 되는 줄기
## kfold를 실험할 최대 k의 수
max_kfold = 15

train_label = train_sorted[:, -1]
train_sorted = train_sorted[:, :-1]

## kf를 2부터 max_kfold까지 늘려가며 에러율을 계산
for kf in range(2, max_kfold + 1):
    p0, p1, m0, m1, s0, s1, err = kfold(train_sorted, train_label, kf, train_zero2one, len(train_sorted))
    record_pi.append([p0, p1])
    record_mu.append([m0, m1])
    record_sig.append([s0, s1])
    record_err.append(err)

## k-fold의 에러율 그래프
plt.figure(figsize=(8, 6))
plt.plot(range(2, max_kfold + 1), record_err, marker='o', linestyle='-', color='b')
plt.title('Error Rate vs Number of GMM Components')
plt.xlabel('Number of GMM Components')
plt.ylabel('Error Rate')
plt.xticks(range(2, max_kfold + 1))
plt.grid(True)
plt.show()
print(record_err)

# 최적의 k-fold를 바탕으로 테스트 데이터셋을 처리
min_idx = np.argmin(record_err)
final_err = cal_test_err(test[:, :-1], test_label, record_pi[min_idx], record_mu[min_idx], record_sig[min_idx])

## 최종 오류율 출력
print(final_err)
