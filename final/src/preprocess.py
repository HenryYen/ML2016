import numpy as np
# from count_frequency.py


def preprocess(line):
    parts = line.rstrip().split(',')
    parts[1] = map_col1(parts[1])
    parts[2] = map_col2(parts[2])
    parts[3] = map_col3(parts[3])
    if len(parts) == 42:
        parts[41] = map_type(parts[41])
    return [float(e) for e in parts]
    

def map_col1(raw):
    category = ['udp', 'icmp', 'tcp']    
    if raw not in category:
        print 'MAP_COL1_ERROR'   
    return category.index(raw)
    
def map_col2(raw):
    category = ['aol', 'urp_i', 'netbios_ssn', 'Z39_50', 'smtp', 'domain', 'private', 'echo', 'printer', 'red_i', 'eco_i', 'ftp_data', 'sunrpc', 'urh_i', 'uucp', 'pop_3', 'pop_2', 'systat', 'ftp', 'sql_net', 'whois', 'tftp_u', 'netbios_dgm', 'efs', 'remote_job', 'daytime', 'pm_dump', 'other', 'finger', 'ldap', 'netbios_ns', 'kshell', 'iso_tsap', 'ecr_i', 'nntp', 'http_2784', 'shell', 'domain_u', 'uucp_path', 'courier', 'exec', 'tim_i', 'netstat', 'telnet', 'gopher', 'rje', 'hostnames', 'link', 'ssh', 'http_443', 'csnet_ns', 'X11', 'IRC', 'harvest', 'login', 'icmp', 'supdup', 'name', 'nnsp', 'mtp', 'http', 'ntp_u', 'bgp', 'ctf', 'klogin', 'vmnet', 'time', 'discard', 'imap4', 'auth', 'http_8001']
    if raw not in category:
        print 'MAP_COL2_ERROR'   
    return category.index(raw)
    
def map_col3(raw):
    category = ['OTH', 'RSTR', 'S3', 'S2', 'S1', 'S0', 'RSTOS0', 'REJ', 'SH', 'RSTO', 'SF']
    if raw not in category:
        print 'MAP_COL3_ERROR'   
    return category.index(raw)


def map_type(raw_label):
    attack = [['normal.'], ['apache2.', 'back.', 'mailbomb.', 'processtable.', 'snmpgetattack.', 'teardrop.', 'smurf.', 'land.', 'neptune.', 'pod.', 'udpstorm.'], ['ps.', 'buffer_overflow.', 'perl.', 'rootkit.', 'loadmodule.', 'xterm.', 'sqlattack.', 'httptunnel.'], ['ftp_write.', 'guess_passwd.', 'snmpguess.', 'imap.', 'spy.', 'warezclient.', 'warezmaster.', 'multihop.', 'phf.', 'imap.', 'named.', 'sendmail.', 'xlock.', 'xsnoop.', 'worm.'], ['nmap.', 'ipsweep.', 'portsweep.', 'satan.', 'mscan.', 'saint.', 'worm.']]   
    for i in attack:
        if raw_label in i:
            return attack.index(i)
    print 'MAP_TYPE_ERROR' 

            



    
            
            
