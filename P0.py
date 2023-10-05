import numpy as np


##

def entropy_binary(p0):
    p1 = 1 - p0 #calculates the complement probability
    probabilities = [p0, p1]
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy


##

def joint_entropy(pmf):

    entropy = 0.0
    for p in pmf.values():
        entropy -= p * np.log2(p)

    return entropy

##

def conditional_entropy(joint_pmf, marginal_pmf):

    entropy = 0.0
    for (x, y), p_joint in joint_pmf.items():
        p_x = marginal_pmf[x]
        if p_x > 0:
            entropy -= p_joint * np.log2(p_joint / p_x)

    return entropy


##

def mutual_information(joint_pmf, marginal_pmf_x, marginal_pmf_y):

    mi = 0
    for (x,y), p_joint in joint_pmf.items():

        p_x = marginal_pmf_x[x]
        p_y = marginal_pmf_y[y]

        if (p_x>0) and (p_y>0):
            mi += (p_joint * np.log2(p_joint/(p_x*p_y)))

    return mi


##


def normalized_conditional_entropy(joint_pmf, marginal_pmf):

    entropy = 0.0
    for (x, y), p_joint in joint_pmf.items():
        p_x = marginal_pmf[x]
        ent_x = joint_entropy(marginal_pmf)
        if p_x > 0:
            entropy -= (p_joint * np.log2(p_joint / p_x)) / ent_x

    return entropy


def normalized_joint_entropy(joint_pmf, marginal_pmf_x, marginal_pmf_y):

    entropy = 0.0
    for p in joint_pmf.values():

        ent_x = joint_entropy(marginal_pmf_x)
        ent_y = joint_entropy(marginal_pmf_y)
        entropy -= p * np.log2(p) / (ent_x + ent_y)

    return entropy


def normalized_mutual_information(joint_pmf, marginal_pmf_x, marginal_pmf_y):

    n_joint_entropy = normalized_joint_entropy(joint_pmf, marginal_pmf_x, marginal_pmf_y)
    normalized_mi = (1/n_joint_entropy) - 1
    return normalized_mi

##

joint_pmf = {   #dictionary with keys as variables and values as probability
    (1, 1): 0.25,
    (1, 2): 0.10,
    (2, 1): 0.15,
    (2, 2): 0.50
}


marginal_pmf_x = {  #marginal pmf pf x
    1: 0.35,
    2: 0.65
}

marginal_pmf_y = {   #marginal pmf of y
    1: 0.40,
    2: 0.60
}


j_entropy = joint_entropy(joint_pmf)
print("Joint entropy:", j_entropy)

c_entropy = conditional_entropy(joint_pmf, marginal_pmf_x)
print("Conditional entropy:", c_entropy)

mi = mutual_information(joint_pmf, marginal_pmf_x, marginal_pmf_y)
print("Mutual information: ", mi)

j_entropy_norm = normalized_joint_entropy(joint_pmf, marginal_pmf_x, marginal_pmf_y)
print("Normalized Joint entropy:", j_entropy_norm)

c_entropy_norm = normalized_conditional_entropy(joint_pmf, marginal_pmf_x)
print("Normalized Conditional entropy:", c_entropy_norm)

mi_norm = normalized_mutual_information(joint_pmf, marginal_pmf_x, marginal_pmf_y)
print("Normalized Mutual information: ", mi_norm)

# print("My norm conditional: ", conditional_entropy(joint_pmf,marginal_pmf_x)/joint_entropy(marginal_pmf_x))

