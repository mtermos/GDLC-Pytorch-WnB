class DatasetInfo:
    def __init__(
            self,
            name,
            path,
            file_type,

            # Key Columns names
            src_ip_col,
            src_port_col,
            dst_ip_col,
            dst_port_col,
            flow_id_col,
            timestamp_col,
            label_col,
            class_col,

            class_num_col=None,
            timestamp_format=None,

            centralities_set=1,

            # Columns to be dropped from the dataset during preprocessing.
            drop_columns=[],

            # Columns to be dropped from the dataset during preprocessing.
            weak_columns=[],
    ):

        self.name = name
        self.path = path
        self.file_type = file_type
        self.src_ip_col = src_ip_col
        self.src_port_col = src_port_col
        self.dst_ip_col = dst_ip_col
        self.dst_port_col = dst_port_col
        self.flow_id_col = flow_id_col
        self.timestamp_col = timestamp_col
        self.timestamp_format = timestamp_format
        self.label_col = label_col
        self.class_col = class_col
        self.centralities_set = centralities_set
        self.class_num_col = class_num_col
        self.drop_columns = drop_columns
        self.weak_columns = weak_columns


datasets_list = [
    DatasetInfo(name="cic_ton_iot_5_percent",
                path="datasets/cic_ton_iot_5_percent/cic_ton_iot_5_percent.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="%d/%m/%Y %I:%M:%S %p",

                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Cnt', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Subflow Bwd Pkts', 'Flow IAT Mean', 'Fwd Pkt Len Max', 'Flow IAT Max', 'Active Std', 'Bwd Header Len', 'Tot Bwd Pkts', 'Bwd Pkt Len Mean', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
                              'CWE Flag Count', 'Bwd IAT Tot', 'Fwd IAT Mean', 'Fwd Pkt Len Std', 'Pkt Len Mean', 'Flow IAT Min', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max', 'Pkt Len Var', 'FIN Flag Cnt', 'Bwd IAT Mean', 'Idle Mean', 'Pkt Len Max', 'Flow Pkts/s', 'Flow Duration', 'Pkt Len Std', 'Fwd IAT Tot', 'PSH Flag Cnt', 'Active Mean', 'Bwd Pkt Len Std', 'Fwd Pkt Len Mean']
                ),
    DatasetInfo(name="cic_ton_iot",
                path="datasets/cic_ton_iot/cic_ton_iot.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="%d/%m/%Y %I:%M:%S %p",

                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Cnt', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Subflow Bwd Pkts', 'Active Mean', 'Active Std', 'Bwd Header Len', 'Bwd IAT Mean', 'Bwd IAT Tot', 'Bwd Pkt Len Max', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Bwd Seg Size Avg', 'CWE Flag Count', 'FIN Flag Cnt',
                              'Flow Duration', 'Flow IAT Max', 'Flow IAT Mean', 'Flow IAT Min', 'Flow Pkts/s', 'Fwd IAT Mean', 'Fwd IAT Tot', 'Fwd Pkt Len Max', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Fwd Seg Size Avg', 'Idle Mean', 'PSH Flag Cnt', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'Pkt Size Avg', 'Tot Bwd Pkts', 'TotLen Bwd Pkts']
                ),
    DatasetInfo(name="cic_ids_2017_5_percent",
                path="datasets/cic_ids_2017_5_percent/cic_ids_2017_5_percent.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                centralities_set=2,
                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Fwd IAT Min',  'Idle Max', 'Flow IAT Mean',  'Protocol',   'Fwd Pkt Len Max', 'Flow IAT Max', 'Active Std', 'Subflow Fwd Pkts', 'Bwd Pkt Len Mean', 'Tot Bwd Pkts', 'Pkt Size Avg',
                              'Subflow Bwd Pkts', 'Bwd IAT Std', 'Fwd IAT Mean', 'Fwd Pkt Len Std', 'Pkt Len Mean', 'Flow IAT Std', 'Fwd URG Flags', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max',  'Pkt Len Var',  'Tot Fwd Pkts', 'Bwd IAT Mean', 'TotLen Fwd Pkts', 'Fwd PSH Flags', 'Idle Mean', 'Pkt Len Max', 'Flow Pkts/s', 'Flow Duration', 'Pkt Len Std', 'Fwd IAT Max',  'Fwd IAT Tot', 'RST Flag Cnt', 'Subflow Bwd Byts', 'Active Mean', 'Bwd Pkt Len Std', 'Fwd Pkt Len Mean']
                ),
    DatasetInfo(name="cic_ids_2017",
                path="datasets/cic_ids_2017/cic_ids_2017.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Fwd IAT Min',  'Idle Max', 'Flow IAT Mean',  'Protocol',   'Fwd Pkt Len Max', 'Flow IAT Max', 'Active Std', 'Subflow Fwd Pkts', 'Bwd Pkt Len Mean', 'Tot Bwd Pkts', 'Pkt Size Avg',
                              'Subflow Bwd Pkts', 'Bwd IAT Std', 'Fwd IAT Mean', 'Fwd Pkt Len Std', 'Pkt Len Mean', 'Flow IAT Std', 'Fwd URG Flags', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max',  'Pkt Len Var',  'Tot Fwd Pkts', 'Bwd IAT Mean', 'TotLen Fwd Pkts', 'Fwd PSH Flags', 'Idle Mean', 'Pkt Len Max', 'Flow Pkts/s', 'Flow Duration', 'Pkt Len Std', 'Fwd IAT Max',  'Fwd IAT Tot', 'RST Flag Cnt', 'Subflow Bwd Byts', 'Active Mean', 'Bwd Pkt Len Std', 'Fwd Pkt Len Mean']
                ),
    DatasetInfo(name="cic_bot_iot",
                path="datasets/cic_bot_iot/cic_bot_iot.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Active Max', 'Active Mean', 'Bwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Header Len', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Tot', 'Bwd PSH Flags', 'Bwd Pkt Len Max', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Min', 'Bwd Pkt Len Std', 'Bwd Pkts/b Avg', 'Bwd URG Flags', 'CWE Flag Count', 'Flow Duration', 'Flow IAT Max', 'Flow IAT Mean', 'Flow IAT Std', 'Flow Pkts/s', 'Fwd Blk Rate Avg',
                              'Fwd Byts/b Avg', 'Fwd Header Len', 'Fwd IAT Max', 'Fwd IAT Mean', 'Fwd PSH Flags', 'Fwd Pkt Len Max', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Min', 'Fwd Pkts/b Avg', 'Fwd Seg Size Min', 'Fwd URG Flags', 'Idle Max', 'Idle Mean', 'Init Fwd Win Byts', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Size Avg', 'Subflow Fwd Pkts', 'Tot Bwd Pkts', 'Tot Fwd Pkts', 'TotLen Bwd Pkts', 'TotLen Fwd Pkts']
                ),
    DatasetInfo(name="cic_ton_iot_modified",
                path="datasets/cic_ton_iot_modified/cic_ton_iot_modified.parquet",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['ACK Flag Cnt', 'Active Mean', 'Active Std', 'Bwd Byts/b Avg', 'Bwd Header Len', 'Bwd IAT Mean', 'Bwd IAT Tot', 'Bwd PSH Flags', 'Bwd Pkt Len Max', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Bwd Seg Size Avg', 'Bwd URG Flags', 'CWE Flag Count', 'ECE Flag Cnt', 'FIN Flag Cnt', 'Flow Duration', 'Flow IAT Max', 'Flow IAT Mean', 'Flow IAT Min', 'Flow IAT Std', 'Flow Pkts/s',
                              'Fwd Blk Rate Avg', 'Fwd Byts/b Avg', 'Fwd Header Len', 'Fwd IAT Mean', 'Fwd IAT Tot', 'Fwd Pkt Len Max', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Fwd Pkts/b Avg', 'Fwd Seg Size Avg', 'Fwd URG Flags', 'Idle Mean', 'PSH Flag Cnt', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'Pkt Size Avg', 'Subflow Bwd Pkts', 'Tot Bwd Pkts', 'Tot Fwd Pkts', 'TotLen Bwd Pkts', 'URG Flag Cnt']
                ),

    DatasetInfo(name="nf_ton_iotv2_modified",
                path="./datasets/nf_ton_iotv2_modified/nf_ton_iotv2_modified.parquet",
                file_type="parquet",
                src_ip_col="IPV4_SRC_ADDR",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="IPV4_DST_ADDR",
                dst_port_col="L4_DST_PORT",
                flow_id_col=None,
                timestamp_col=None,
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["IPV4_SRC_ADDR", "L4_SRC_PORT",
                              "IPV4_DST_ADDR", "L4_DST_PORT", "Attack"],
                weak_columns=['CLIENT_TCP_FLAGS', 'DURATION_IN', 'ICMP_TYPE', 'IN_BYTES', 'LONGEST_FLOW_PKT',
                              'MAX_TTL', 'MIN_TTL', 'PROTOCOL', 'RETRANSMITTED_OUT_BYTES', 'TCP_FLAGS', 'TCP_WIN_MAX_IN'],
                ),

    DatasetInfo(name="ccd_inid_modified",
                path="./datasets/ccd_inid_modified/ccd_inid_modified.parquet",
                file_type="parquet",
                src_ip_col="src_ip",
                src_port_col="src_port",
                dst_ip_col="dst_ip",
                dst_port_col="dst_port",
                flow_id_col="id",
                timestamp_col=None,
                label_col="traffic_type",
                class_col="atk_type",
                class_num_col="Class",
                timestamp_format=None,
                drop_columns=["id", "src_ip", "src_port",
                              "dst_ip", "dst_port", "atk_type", 'Unnamed: 0', 'src_ip_is_private', 'dst_ip_is_private',
                              'expiration_id', 'splt_direction', 'splt_ps', 'splt_piat_ms'],
                weak_columns=['application_name', 'bidirectional_ack_packets', 'bidirectional_bytes', 'bidirectional_cwr_packets', 'bidirectional_duration_ms', 'bidirectional_ece_packets', 'bidirectional_fin_packets', 'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms', 'bidirectional_max_piat_ms', 'bidirectional_mean_piat_ms', 'bidirectional_mean_ps', 'bidirectional_min_piat_ms', 'bidirectional_packets', 'bidirectional_psh_packets', 'bidirectional_rst_packets', 'bidirectional_stddev_piat_ms', 'bidirectional_stddev_ps', 'bidirectional_syn_packets',
                              'bidirectional_urg_packets', 'dst2src_bytes', 'dst2src_cwr_packets', 'dst2src_duration_ms', 'dst2src_ece_packets', 'dst2src_first_seen_ms', 'dst2src_last_seen_ms', 'dst2src_max_ps', 'dst2src_mean_ps', 'dst2src_min_piat_ms', 'dst2src_packets', 'dst2src_stddev_piat_ms', 'dst2src_stddev_ps', 'dst2src_urg_packets', 'ip_version', 'src2dst_bytes', 'src2dst_cwr_packets', 'src2dst_duration_ms', 'src2dst_ece_packets', 'src2dst_first_seen_ms', 'src2dst_mean_ps', 'src2dst_min_piat_ms', 'src2dst_packets', 'src2dst_syn_packets', 'src2dst_urg_packets', 'vlan_id'],
                ),

    DatasetInfo(name="nf_uq_nids_modified",
                path="./datasets/nf_uq_nids_modified/nf_uq_nids_modified.parquet",
                file_type="parquet",
                src_ip_col="IPV4_SRC_ADDR",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="IPV4_DST_ADDR",
                dst_port_col="L4_DST_PORT",
                flow_id_col=None,
                timestamp_col=None,
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["IPV4_SRC_ADDR", "L4_SRC_PORT",
                              "IPV4_DST_ADDR", "L4_DST_PORT", "Attack", "Dataset"],
                weak_columns=[],
                ),

    DatasetInfo(name="edge_iiot",
                path="./datasets/edge_iiot/edge_iiot.parquet",
                file_type="parquet",
                src_ip_col="ip.src_host",
                src_port_col=None,
                dst_ip_col="ip.dst_host",
                dst_port_col=None,
                flow_id_col=None,
                timestamp_col="frame.time",
                label_col="Attack_label",
                class_col="Attack_type",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["ip.src_host", "ip.dst_host", "http.tls_port",
                              "dns.qry.type", "mqtt.msg_decoded_as", "frame.time", "Attack_type"],
                weak_columns=["tcp.flags", "mqtt.conflags", "mqtt.conflag.cleansess", "mbtcp.trans_id", "mqtt.hdrflags", "mqtt.msg",
                              "mqtt.len", "dns.retransmit_request", "http.request.method", "icmp.unused", "mbtcp.len", "mqtt.proto_len", "arp.opcode"]
                ),

    DatasetInfo(name="nf_cse_cic_ids2018",
                path="./datasets/nf_cse_cic_ids2018/nf_cse_cic_ids2018.parquet",
                file_type="parquet",
                src_ip_col="IPV4_SRC_ADDR",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="IPV4_DST_ADDR",
                dst_port_col="L4_DST_PORT",
                flow_id_col=None,
                timestamp_col=None,
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["IPV4_SRC_ADDR", "L4_SRC_PORT",
                              "IPV4_DST_ADDR", "L4_DST_PORT", "Attack"],
                weak_columns=['IN_BYTES', 'OUT_BYTES']
                ),

    DatasetInfo(name="nf_bot_iotv2",
                path="./datasets/nf_bot_iotv2/nf_bot_iotv2.parquet",
                file_type="parquet",
                src_ip_col="IPV4_SRC_ADDR",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="IPV4_DST_ADDR",
                dst_port_col="L4_DST_PORT",
                flow_id_col=None,
                timestamp_col=None,
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["IPV4_SRC_ADDR", "L4_SRC_PORT",
                              "IPV4_DST_ADDR", "L4_DST_PORT", "Attack"],
                weak_columns=["RETRANSMITTED_IN_BYTES", "MIN_TTL", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "PROTOCOL", "SERVER_TCP_FLAGS", "CLIENT_TCP_FLAGS",
                              "LONGEST_FLOW_PKT", "NUM_PKTS_512_TO_1024_BYTES", "RETRANSMITTED_OUT_BYTES", "IN_PKTS", "TCP_FLAGS", "IN_BYTES", "ICMP_TYPE"]
                ),

    DatasetInfo(name="nf_uq_nids",
                path="./datasets/nf_uq_nids/nf_uq_nids.parquet",
                file_type="parquet",
                src_ip_col="IPV4_SRC_ADDR",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="IPV4_DST_ADDR",
                dst_port_col="L4_DST_PORT",
                flow_id_col=None,
                timestamp_col=None,
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["IPV4_SRC_ADDR", "L4_SRC_PORT",
                              "IPV4_DST_ADDR", "L4_DST_PORT", "Attack", "Dataset"],
                weak_columns=[],
                ),

    DatasetInfo(name="x_iiot",
                path="./datasets/x_iiot/x_iiot.parquet",
                file_type="parquet",
                src_ip_col="Scr_IP",
                src_port_col="Scr_port",
                dst_ip_col="Des_IP",
                dst_port_col="Des_port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="class3",
                class_col="class2",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["Scr_IP", "Scr_port", "Des_IP",
                              "Des_port", "Timestamp", "Date", "class1", "class2"],
                weak_columns=["Process_activity", "Login_attempt", "is_syn_only", "Avg_iowait_time", "Avg_num_Proc/s", "Duration", "Des_pkts_ratio",
                              "is_SYN_with_RST", "Succesful_login", "OSSEC_alert", "is_pure_ack", "Conn_state", "Bad_checksum", "File_activity", "Avg_rtps", "Is_SYN_ACK"]
                ),
]


datasets = {dataset.name: dataset for dataset in datasets_list}

cn_measures = [
    ["betweenness", "local_betweenness", "degree", "local_degree",
     "eigenvector", "closeness", "pagerank", "local_pagerank", "k_core", "k_truss", "Comm"],
    ["betweenness", "global_betweenness", "degree", "global_degree",
     "eigenvector", "closeness", "pagerank", "global_pagerank", "k_core", "k_truss", "mv"],
    ["betweenness", "local_betweenness", "pagerank",
        "local_pagerank", "k_core", "k_truss", "Comm"],
    ["betweenness", "global_betweenness", "pagerank",
        "global_pagerank", "k_core", "k_truss", "mv"]
]

network_features = [
    ['src_betweenness', 'dst_betweenness', 'src_local_betweenness', 'dst_local_betweenness', 'src_degree', 'dst_degree', 'src_local_degree', 'dst_local_degree', 'src_eigenvector',
     'dst_eigenvector', 'src_closeness', 'dst_closeness', 'src_pagerank', 'dst_pagerank', 'src_local_pagerank', 'dst_local_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_Comm', 'dst_Comm'],
    ['src_betweenness', 'dst_betweenness', 'src_global_betweenness', 'dst_global_betweenness', 'src_degree', 'dst_degree', 'src_global_degree', 'dst_global_degree', 'src_eigenvector',
     'dst_eigenvector', 'src_closeness', 'dst_closeness', 'src_pagerank', 'dst_pagerank', 'src_global_pagerank', 'dst_global_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_mv', 'dst_mv'],
    ['src_betweenness', 'dst_betweenness', 'src_local_betweenness', 'dst_local_betweenness', 'src_pagerank',
     'dst_pagerank', 'src_local_pagerank', 'dst_local_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_Comm', 'dst_Comm'],
    ['src_betweenness', 'dst_betweenness', 'src_global_betweenness', 'dst_global_betweenness', 'src_pagerank',
     'dst_pagerank', 'src_global_pagerank', 'dst_global_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_mv', 'dst_mv'],

]
