def logme(log, x, P, t, u):
    log['Xs'].append(x)
    log['Ps'].append(P)
    log['ts'].append(t)
    log['us'].append(u)
    return log