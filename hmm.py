#!/usr/bin/env python

"""
#===================================#
# Problem Set 5: HMM Implementation #
# Michael Ting                      #
# CS176 Fall 2013                   #
# 9 December 2013                   #
#===================================#
"""

from math import *
from argparse import ArgumentParser

def process_fasta(infile):
    """
    A generator that grabs all sequences from a FASTA file
    """
    f = open(infile, "r")
    
    name, seq = None, []
    for line in f:
        if line.startswith(">"):
            # when we get to the next sequence, yield the one before it
            # seq is a list, so we joing everything into one string
            if name: 
                yield (name, ''.join(seq))
            name, seq = line.strip(), []    # parsing starts here
        # sequence spanning multiple lines
        else:
            seq.append(line.strip())
    # handles the last sequence in the file
    if name: 
        yield (name, ''.join(seq))
        
    f.close()
        
def process_params(infile):
    """
    Given a file of parameters, return:
    - a table of marginal probabilities
        - indexed by k
        - k = {1,2,3,4}
    - a table of transition probabilities
        - [k][l] index, where k is row, l is column
        - k = {1,2,3,4}
    - a table of emission probabilities
        - [k][0] = P(I|Q=k)
        - [k][1] = P(D|Q=k)
        - k = {1,2,3,4}
    """
    marg = {}   # {k:P(Q_1=k),...
                # }
    trans = {}  # {k:{l:a_{kl},
                #     l:a_{kl}},
                #  k:{l:a_{kl},
                #     l:a_{kl}},...
                # }
    emis = {}   # {k:{I:prob,
                #     D:prob}},...
                # }
    f = open(infile, "r")
    typecount = -1
    transcount = 1
    commentflag = False
    for line in f:
        if line.startswith("#"):
            if not commentflag:
                typecount += 1
                commentflag = True
            else:
                pass
        elif line.isspace():
            continue
        else:
            commentflag = False
            if typecount == 0:  # marginals
                margrow = line.split()
                k = int(margrow[0])
                p = float(margrow[1])
                marg[k] = p
            elif typecount == 1:    # transitions
                transrow = line.split()
                l_indexed = {}
                for ind in range(len(transrow)):
                    # ind+1 because params file has bad transition probs formatting!! GRR.
                    l_indexed[int(ind+1)] = float(transrow[ind])
                trans[transcount] = l_indexed
                transcount += 1
            elif typecount == 2:    # emissions
                emisrow = line.split()
                k = int(emisrow[0])
                p_i = float(emisrow[1])
                p_d = float(emisrow[2])
                obstable = {}
                obstable['I'] = p_i
                obstable['D'] = p_d
                emis[k] = obstable
            else:
                raise Exception("Badly formatted file!")
    
    f.close()
    
    return marg, trans, emis
        
def observed(a, b):
    """
    Creates a string of observed symbols given two input strings
    symbols are {I,D}, where "I" corresponds to identical and
    "D" corresponds to different
    """
    if len(a) != len(b):
        raise Exception("sequences are not of the same length!")
    
    obs = ""
    for i in range(len(a)):
        if a[i] == b[i]:
            obs += "I"
        else:
            obs += "D"
    
    return obs
    
def get_obs(infile):
    """
    Returns the observed string given an input FASTA
    file with two sequences of identical length
    """
    seqs = process_fasta(infile)
    seqlst = []
    for name, seq in seqs:
        seqlst.append(seq)
    seq1 = seqlst[0]
    seq2 = seqlst[1]
    obs = observed(seq1, seq2)
    return obs
    
def write_LLfile(initLL, estLL, outfile):
    """
    Write the log likelihood file
    """
    f = open(outfile, 'w')
    f.write("# Likelihood under {initial, estimated} parameters\n")
    f.write("%.6f\n" % initLL)
    f.write("%.6f\n" % estLL)
    f.flush()
    f.close()
    print "Log likelihoods written to %s" % str(outfile)
    return
    
###############################################################################
# Forward Algorithm                                                           #
###############################################################################
    
def forward_LL(seqfile, paramfile):
    """
    Given input files:
    - sequences (FASTA)
    - parameters
    Calculate the log likelihood using the forward algorithm
    """
    obs = get_obs(seqfile)
    marg,trans,emis = process_params(paramfile)
    loglikelihood = forward(obs, marg, trans, emis)
    return float('%.6f' % loglikelihood)
    
def forward(obs, marg, trans, emis):
    """
    Calculation of f_t(k), the forward algorithm
    Returns the log likelihood P(x_{0:L-1}|theta)
    """
    ftable = forward_table(obs, marg, trans, emis)
    end = ftable[len(obs)-1]
    
    # sum of exponents calculation correction
    Dval = float("-inf")
    QStates = [1,2,3,4]
    for k in QStates:
        Dval = max(Dval, end[k])
    # sum over all k states at L
    endsum = 0.0
    for k in QStates:
        d_i = end[k] - Dval
        endsum += exp(d_i)
        
    logendsum = log(endsum) + Dval
    
    return logendsum
    
def forward_table(obs, marg, trans, emis):
    """
    Calculation of f_t(k), the forward algorithm
    Returns a table of log f_t(k) values
    """
    # index by  {t: {k:f_t{k},
    #                k:f_t{k}},
    #            t: {k:f_t{k}},...}
    # ftable[t][k] --> f_t{k}
    ftable = {}
    # initialization
    log_f0_1 = log(emis[1][obs[0]]) + log(marg[1])
    log_f0_2 = log(emis[2][obs[0]]) + log(marg[2])
    log_f0_3 = log(emis[3][obs[0]]) + log(marg[3])
    log_f0_4 = log(emis[4][obs[0]]) + log(marg[4])
    fk = {}
    fk[1] = log_f0_1
    fk[2] = log_f0_2
    fk[3] = log_f0_3
    fk[4] = log_f0_4
    ftable[0] = fk
    # begin the recursion
    QStates = [1,2,3,4]
    for index in range(1,len(obs)):
        fj = {}
        for j in QStates:
            fj[j] = forward_recursion(index, j, obs, trans, emis, ftable)
        ftable[index] = fj
    
    return ftable

def forward_recursion(t, j, obs, trans, emis, ftable):
    """
    The recursion step of the forward algorithm
    Returns the log space value f_t{j}
    """    
    # compute the D value to prevent overflow
    Dval = float("-inf")
    QStates = [1,2,3,4]
    for i in QStates:
        Dval = max(Dval, (ftable[t-1][i] + log(trans[i][j])) )
    
    # compute the sum using the D value
    fsum = 0.0
    for i in QStates:
        d_i = ftable[t-1][i] + log(trans[i][j])
        fsum += exp(d_i - Dval)
        
    logfsum = log(fsum)
    
    log_ftj = log(emis[j][obs[t]]) + logfsum + Dval
    
    return log_ftj
    
###############################################################################
# Backward Algorithm                                                          #
###############################################################################

def backward_LL(seqfile, paramfile):
    """
    Given input files:
    - sequences (FASTA)
    - parameters
    Calculate the log likelihood using the forward algorithm
    """
    obs = get_obs(seqfile)
    marg,trans,emis = process_params(paramfile)
    loglikelihood = backward(obs, marg, trans, emis)
    return float('%.6f' % loglikelihood)

def backward(obs, marg, trans, emis):
    """
    Calculation of b_t(k), the forward algorithm
    Returns the log likelihood P(x_{0:L-1}|theta)
    """
    btable = backward_table(obs, trans, emis)
    end = btable[0]
    
    # sum of exponents calculation correction
    Dval = float("-inf")
    QStates = [1,2,3,4]
    for k in QStates:
        Dval = max(Dval, (log(emis[k][obs[0]]) + end[k] + log(marg[k])) )
    # sum over all k states at L
    endsum = 0.0
    for k in QStates:
        d_i = log(emis[k][obs[0]]) + end[k] + log(marg[k]) - Dval
        endsum += exp(d_i)
        
    logendsum = log(endsum) + Dval
    
    return logendsum
    
def backward_table(obs, trans, emis):
    """
    Calculation of b_t(k), the backward algorithm
    Returns a table of log b_t(k) values
    """
    # index by  {t: {k:f_t{k},
    #                k:f_t{k}},
    #            t: {k:f_t{k}},...}
    # btable[t][k] --> f_t{k}
    btable = {}
    # initialization
    log_bL_1 = log(1)
    log_bL_2 = log(1)
    log_bL_3 = log(1)
    log_bL_4 = log(1)
    bk = {}
    bk[1] = log_bL_1
    bk[2] = log_bL_2
    bk[3] = log_bL_3
    bk[4] = log_bL_4
    btable[len(obs)-1] = bk
    # begin the recursion
    QStates = [1,2,3,4]
    for index in reversed(range(0,len(obs)-1)):
        bi = {}
        for i in QStates:
            bi[i] = backward_recursion(index, i, obs, trans, emis, btable)
        btable[index] = bi
    
    return btable
    
def backward_recursion(t, i, obs, trans, emis, btable):
    """
    The recursion step of the forward algorithm
    Returns the log space value f_t{j}
    """    
    # compute the D value to prevent overflow
    Dval = float("-inf")
    QStates = [1,2,3,4]
    for j in QStates:
        Dval = max(Dval, (log(trans[i][j]) + log(emis[j][obs[t+1]]) + btable[t+1][j]))
    
    # compute the sum using the D value
    bsum = 0.0
    for j in QStates:
        d_j = log(trans[i][j]) + log(emis[j][obs[t+1]]) + btable[t+1][j]
        bsum += exp(d_j - Dval)
        
    logbsum = log(bsum)
    
    log_btk = logbsum + Dval
    
    return log_btk

###############################################################################
# Expectation-Maximization Algorithm                                                          #
###############################################################################

def baum_welch(seqfile, paramfile, steps):
    """
    Uses Expectation-Maximization to estimate parameters of a model
    """
    obs = get_obs(seqfile)
    marg, trans, emis = process_params(paramfile)
    ftable = forward_table(obs, marg, trans, emis)
    btable = backward_table(obs, trans, emis)
    
    for i in range(steps):
        LL = forward(obs, marg, trans, emis) # log likelihood
        # E-Step
        gamma = gamma_table(ftable, btable, obs, LL)
        epsilon = epsilon_table(ftable, btable, trans, emis, obs, LL)
        delta = delta_table(ftable, btable, obs, LL)
        # M-Step
        marg = new_marg(gamma)
        trans = new_trans(epsilon)
        emis = new_emis(delta)
        # Update the tables
        ftable = forward_table(obs, marg, trans, emis)
        btable = backward_table(obs, trans, emis)
        print "##############################"
        print "######## Iteration %s ########" % str(i+1)
        print "##############################"
        print "marg: %s" % str(marg)
        print "trans: %s" % str(trans)
        print "emis: %s" % str(emis)
        print ""
        
    #write_paramfile(paramoutfile, marg, trans, emis)
    
    LL = forward(obs, marg, trans, emis)
    #print "Log Likelihood: %.6f" % LL
        
    return marg, trans, emis, LL

def write_paramfile(paramoutfile, marg, trans, emis):
    """
    Writes parameters to outfile
    """
    f = open(paramoutfile,'w')
    f.write("# Marginal Probabilities\n")
    f.flush()
    QStates = [1,2,3,4]
    for k in QStates:
        f.write("%.6e\n" % marg[k])
        f.flush()
    f.write("\n")
    f.write("# Transition Probabilities\n")
    f.flush()
    for k in QStates:
        f.write("%.6e\t%.6e\t%.6e\t%.6e\n" % (trans[k][1], trans[k][2], trans[k][3], trans[k][4]))
        f.flush()
    f.write("\n")
    f.write("# Emission Probabilities\n")
    f.flush()
    for k in QStates:
        f.write("%.6e\t%.6e\t\n" % (emis[k]['I'],emis[k]['D']))
        f.flush()
    f.write("\n")
    f.close()
    print "Parameters written to %s" % str(paramoutfile)
    return

def new_marg(gamma):
    """
    Compute the updated marginal (first position, t=0)
    """
    newmarg = {}
    # margset looks like {k : logprob}
    margset = gamma[0]
    QStates = [1,2,3,4]
    # logsumexp correction
    Dval = float("-inf")
    for j in QStates:
        Dval = max(Dval, margset[j])
        
    innersum = 0.0
    for j in QStates:
        innersum += exp(margset[j] - Dval)
    
    logdenom = Dval + log(innersum)
    
    # calculate the updated marginals
    for k, prob in margset.items():
        # return to normal space from log space
        newmarg[k] = exp(prob - logdenom)
        
    return newmarg

def gamma_table(ftable, btable, obs, LL):
    """
    Compute the marginal posterior values across t
    """
    gamma = {}
    QStates = [1,2,3,4]
    
    # for each index in our sequence, compute prob for each k
    for index in range(len(obs)):
        gk = {}
        for k in QStates:
            log_gamma = ftable[index][k] + btable[index][k] - LL
            gk[k] = log_gamma
        gamma[index] = gk
    return gamma
    
def new_trans(epsilon):
    """
    Compute new transition probabilities
    """
    QStates = [1,2,3,4]
    newtrans = {}
    for i in QStates:
        probs = {}
        for j in QStates:
        
            # logsumexp correction
            Dval = float("-inf")
            for k in QStates:
                Dval = max(Dval, epsilon[i][k])
                
            innersum = 0.0
            for k in QStates:
                innersum += exp(epsilon[i][k] - Dval)
            
            logdenom = Dval + log(innersum)
            
            probs[j] = exp(epsilon[i][j] - logdenom)
        newtrans[i] = probs
    return newtrans

def epsilon_table(ftable, btable, trans, emis, obs, LL):
    """
    Compute the transition values (epsilon)
    """
    epsilon = {}
    QStates = [1,2,3,4]
    # for t = 1 to L-1, sum our values to get Aij
    for i in QStates:
        subtable = {}
        for j in QStates:
            
            # logsumexp correction
            Dval = float("-inf")
            for index in range(0,len(obs)-1):
                fti = ftable[index][i]
                aij = log(trans[i][j])
                ejxt1 = log(emis[j][obs[index+1]])
                bt1j = btable[index+1][j]
                power = fti + aij + ejxt1 + bt1j - LL
                Dval = max(Dval, power)
                
            innersum = 0.0
            for index in range(0,len(obs)-1):
                fti = ftable[index][i]
                aij = log(trans[i][j])
                ejxt1 = log(emis[j][obs[index+1]])
                bt1j = btable[index+1][j]
                power = fti + aij + ejxt1 + bt1j - LL
                innersum += exp(power - Dval)                
                
            transval = Dval + log(innersum)
            subtable[j] = transval # leave values in logspace
            
        epsilon[i] = subtable
    return epsilon
    
def new_emis(delta):
    """
    Compute new emissions
    """
    emis = {}
    QStates = [1,2,3,4]
    alphabet = ['I','D']
    for k in QStates:
        obspair = {}
        for observed in alphabet:
            
            # logsumexp correction
            Dval = float("-inf")
            
            for letter in alphabet:
                Dval = max(Dval, delta[k][letter])
                
            innersum = 0.0
            for letter in alphabet:
                innersum += exp(delta[k][letter] - Dval)
            
            logdenom = Dval + log(innersum)
            
            emisval = delta[k][observed] - logdenom
            # convert from logspace to normal space
            obspair[observed] = exp(emisval)
        emis[k] = obspair
    return emis


def delta_table(ftable, btable, obs, LL):
    """
    Compute the table for the new emission estimates
    """
    delta = {}
    QStates = [1,2,3,4]
    for k in QStates:        
        # logsumexp correction       
        Dval = float("-inf")
        Ival = float("-inf")
        for index in range(len(obs)):
            power = ftable[index][k] + btable[index][k] - LL
            if obs[index] == 'I':
                #handle Identicals
                Ival = max(Dval, power)
            elif obs[index] == 'D':
                #handle Differents
                Dval = max(Ival, power)
            else:
                raise Exception("Incorrectly formatted observations!")
        
        Isum = 0.0
        Dsum = 0.0
        for index in range(len(obs)):
            obschar = obs[index]
            d_i = ftable[index][k] + btable[index][k] - LL
            # handle identicals
            if obschar == 'I':
                Isum += exp(d_i - Ival)
            elif obschar == 'D':
                Dsum += exp(d_i - Dval)
            else:
                raise Exception("Incorrecly formatted observations!")
                
        logIprob = Ival + log(Isum)
        logDprob = Dval + log(Dsum)
        
        newobs = {}
        newobs['I'] = logIprob
        newobs['D'] = logDprob
        delta[k] = newobs
    return delta
                
###############################################################################
# Decoding Algorithms                                                         #
###############################################################################

def viterbi_decoding(seqfile, paramfile):
    """
    Compute the Viterbi decoding
    """
    obs = get_obs(seqfile)
    marg, trans, emis = process_params(paramfile)
    vtable, ptrtable, qstar = viterbi_table(obs, marg, trans, emis)
    vit_decpath, vit_path = viterbi_traceback(vtable, ptrtable, qstar, obs)
        
    return vit_decpath, vit_path
    
def est_viterbi_decoding(seqfile, marg, trans, emis):
    """
    Compute the Viterbi decoding
    """
    obs = get_obs(seqfile)
    vtable, ptrtable, qstar = viterbi_table(obs, marg, trans, emis)
    vit_decpath, vit_path = viterbi_traceback(vtable, ptrtable, qstar, obs)
        
    return vit_decpath, vit_path
    
def viterbi_traceback(vtable, ptrtable, qstar, obs):
    """
    Run the traceback of Viterbi to determine the hidden path
    """
    Qpath = {}
    Qpath[len(obs)-1] = qstar
    
    # produced the reversed path from L to 1
    for index in reversed(range(0,len(obs)-1)):
        qt = ptrtable[index+1][Qpath[index+1]]
        
        Qpath[index] = qt
    # get the correct direction
    correctpath = []
    for index in range(0,len(obs)):
        correctpath.append(Qpath[index])
    
    states = {}
    states[1] = 0.32
    states[2] = 1.75
    states[3] = 4.54
    states[4] = 9.40
    
    # convert to the correct states
    decoding = []
    for q in correctpath:
        decoding.append(states[q])
    
    return decoding, correctpath
    
def viterbi_table(obs, marg, trans, emis):
    """
    Calculation of v_t(k), the viterbi algorithm
    Returns a table of log v_t(k) values
    """
    # index by  {t: {k:v_t{k},
    #                k:v_t{k}},
    #            t: {k:v_t{k}},...}
    # vtable[t][k] --> v_t{k}
    vtable = {}
    # index by {t: {j:k},...}
    ptrtable = {}
    # initialization
    log_v0_1 = log(emis[1][obs[0]]) + log(marg[1])
    log_v0_2 = log(emis[2][obs[0]]) + log(marg[2])
    log_v0_3 = log(emis[3][obs[0]]) + log(marg[3])
    log_v0_4 = log(emis[4][obs[0]]) + log(marg[4])
    vk = {}
    vk[1] = log_v0_1
    vk[2] = log_v0_2
    vk[3] = log_v0_3
    vk[4] = log_v0_4
    vtable[0] = vk
    # begin the recursion
    QStates = [1,2,3,4]
    for index in range(1,len(obs)):
        vj = {}
        ptr_tj = {}
        for j in QStates:
            vj[j], ptr_tj[j] = viterbi_recursion(index, j, obs, trans, emis, vtable)
        vtable[index] = vj
        ptrtable[index] = ptr_tj
    
    # max of exponents calculation correction
    Dval = float("-inf")
    for i in QStates:
        Dval = max(Dval, vtable[len(obs)-1][i])  
    
    bestq = {}
    for k in QStates:
        bestq[k] = exp(vtable[len(obs)-1][k] - Dval)
    qstar, prob = max(bestq.iteritems(), key=lambda x:x[1])
    
    return vtable, ptrtable, qstar

def viterbi_recursion(t, j, obs, trans, emis, vtable):
    """
    The recursion step of the viterbi algorithm
    Returns the log space value v_t{j}
    """    
    QStates = [1,2,3,4]
    
    # max of exponents calculation correction
    Dval = float("-inf")
    for i in QStates:
        Dval = max(Dval, vtable[t-1][i] + log(trans[i][j]))    
    
    # Viterbi decoding
    vlst = []
    for i in QStates:
        val = exp(vtable[t-1][i] + log(trans[i][j]) - Dval)
        vlst.append(val)
    log_vmax = log(max(vlst))
    log_vtj = log(emis[j][obs[t]]) + log_vmax + Dval
    
    Qlst = {}
    for i in QStates:
        ptrval = exp(vtable[t-1][i] + log(trans[i][j]) - Dval)
        Qlst[i] = ptrval
    # compute the argmax of the pointer
    ptr_tj, pval = max(Qlst.iteritems(), key=lambda x:x[1])

    return log_vtj, ptr_tj

def posterior_decoding(seqfile, paramfile):
    """
    Compute the posterior decoding
    """
    obs = get_obs(seqfile)
    marg, trans, emis = process_params(paramfile)
    ftable = forward_table(obs, marg, trans, emis)
    btable = backward_table(obs, trans, emis)
    LL = forward(obs, marg, trans, emis)
    
    QStates = [1,2,3,4]
    states = {}
    states[1] = 0.32
    states[2] = 1.75
    states[3] = 4.54
    states[4] = 9.40
    
    Qpath = []
    for index in range(len(obs)):
        kvals = {}
        for k in QStates:
            log_mpp = ftable[index][k] + btable[index][k] - LL
            kvals[k] = exp(log_mpp)
        # returns the argmax k over kvals
        bestk, prob = max(kvals.iteritems(), key=lambda x:x[1])
        Qpath.append(bestk)
        
    decoded = []
    for k in Qpath:
        decoded.append(states[k])
    
    """
    decfile = "testdecode.txt"
    f = open(decfile,"w")
    for state in decoded:
        f.write("%s\n" % str(state))
        f.flush()
    f.close()
    print "wrote decoding to %s" % decfile
    """
        
    return decoded
    
def est_posterior_decoding(seqfile, marg, trans, emis):
    """
    Compute the posterior decoding
    """
    obs = get_obs(seqfile)
    ftable = forward_table(obs, marg, trans, emis)
    btable = backward_table(obs, trans, emis)
    LL = forward(obs, marg, trans, emis)
    
    QStates = [1,2,3,4]
    states = {}
    states[1] = 0.32
    states[2] = 1.75
    states[3] = 4.54
    states[4] = 9.40
    
    Qpath = []
    for index in range(len(obs)):
        kvals = {}
        for k in QStates:
            log_mpp = ftable[index][k] + btable[index][k] - LL
            kvals[k] = exp(log_mpp)
        # returns the argmax k over kvals
        bestk, prob = max(kvals.iteritems(), key=lambda x:x[1])
        Qpath.append(bestk)
        
    decoded = []
    for k in Qpath:
        decoded.append(states[k])
        
    return decoded

###############################################################################
# Posterior Mean                                                              #
###############################################################################

def posterior_mean(seqfile, paramfile, post_dec):
    """
    Compute the posterior mean at every position
    """
    obs = get_obs(seqfile)
    marg, trans, emis = process_params(paramfile)
    ftable = forward_table(obs, marg, trans, emis)
    btable = backward_table(obs, trans, emis)
    LL = forward(obs, marg, trans, emis) # log likelihood
    
    QStates = [1,2,3,4]
    states = {}
    states[1] = 0.32
    states[2] = 1.75
    states[3] = 4.54
    states[4] = 9.40
    
    pos_mean = []
    for index in range(len(post_dec)):
        
        Dval = float("-inf")
        for k in QStates:
            tmrca = log(states[k])
            ftk = ftable[index][k]
            btk = btable[index][k]
            Dval = max(Dval, tmrca + ftk + btk - LL)   
        
        meansum = 0.0
        for k in QStates:
            tmrca = log(states[k])
            ftk = ftable[index][k]
            btk = btable[index][k]
            log_total = tmrca + ftk + btk - LL
            meansum += exp(log_total - Dval)
        pos_mean.append(exp(Dval + log(meansum)) )
        
    return pos_mean

def est_posterior_mean(seqfile, marg, trans, emis, post_dec):
    """
    Compute the posterior mean at every position
    """
    obs = get_obs(seqfile)
    ftable = forward_table(obs, marg, trans, emis)
    btable = backward_table(obs, trans, emis)
    LL = forward(obs, marg, trans, emis) # log likelihood
    
    QStates = [1,2,3,4]
    states = {}
    states[1] = 0.32
    states[2] = 1.75
    states[3] = 4.54
    states[4] = 9.40
    
    pos_mean = []
    for index in range(len(post_dec)):
        
        Dval = float("-inf")
        for k in QStates:
            tmrca = log(states[k])
            ftk = ftable[index][k]
            btk = btable[index][k]
            Dval = max(Dval, tmrca + ftk + btk - LL)   
        
        meansum = 0.0
        for k in QStates:
            tmrca = log(states[k])
            ftk = ftable[index][k]
            btk = btable[index][k]
            log_total = tmrca + ftk + btk - LL
            meansum += exp(log_total - Dval)
        pos_mean.append(exp(Dval + log(meansum)) )
        
    return pos_mean

def write_decodings(outfile, vit, pos, pmean):
    """
    Write all decodings to a new file
    """
    f = open(outfile,"w")
    f.write("# Viterbi_decoding posterior_decoding posterior_mean\n")
    f.flush()
    for i in range(len(vit)):
        line = "%s\t%s\t%.6f\n" % (vit[i], pos[i], pmean[i])
        f.write(line)
        f.flush()
    f.close()
    print "wrote decoding to %s" % outfile
    return

def main():
    """
    Executes when run from the command line
    """
    parser = ArgumentParser(description="Set parameters for calculation")
    
    parser.add_argument("infile", metavar="in", help="Input FASTA sequence file")
    parser.add_argument("mutname", metavar="out", help="Mutation rate for output file, e.g. mu, 4mu")
    parser.add_argument("parameters", metavar="param", help="Input parameters file")
    
    args = parser.parse_args()
    
    infile = args.infile
    mutname = args.mutname
    paramfile = args.parameters
    print "Computing parameters and likelihoods..."
    # parameter estimation and likelihoods
    EMsteps = 15
    initLL = forward_LL(infile, paramfile)
    marg, trans, emis, estLL = baum_welch(infile, paramfile, EMsteps)
    print "Running initial parameter decoding algorithms..."
    # decoding
    vit_decoding, path = viterbi_decoding(infile, paramfile)
    pos_decoding = posterior_decoding(infile, paramfile)
    pos_mean = posterior_mean(infile, paramfile, path)
    print "Running estimated parameter decoding algorithms..."
    # estimated decoding
    evd, estpath = est_viterbi_decoding(infile, marg, trans, emis)
    est_posdec = est_posterior_decoding(infile, marg, trans, emis)
    est_posmean = est_posterior_mean(infile, marg, trans, emis, estpath)
    
    print "Initial Log Likelihood: %.6f" % initLL
    print "Estimated Log Likelihood: %.6f" % estLL
    print ""
    paramoutfile = './estimated_parameters_%s.txt' % mutname
    print "Writing %s..." % paramoutfile
    write_paramfile(paramoutfile, marg, trans, emis)
    LLoutfile = './likelihoods_%s.txt' % mutname
    print "Writing %s..." % LLoutfile
    write_LLfile(initLL, estLL, LLoutfile)
    initdecfile = './decodings_initial_%s.txt' % mutname
    print "Writing %s..." % initdecfile
    write_decodings(initdecfile, vit_decoding, pos_decoding, pos_mean)
    estdecfile = './decodings_estimated_%s.txt' % mutname
    print "Writing %s..." % estdecfile
    write_decodings(estdecfile, evd, est_posdec, est_posmean)
    print "Program complete."
        
if __name__ == "__main__":
    main()