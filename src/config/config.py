MODULES_TO_INSTRUMENT = ["torch"]
INCLUDED_WRAP_LIST = ["Net", "DataParallel"]  # FIXME: Net & DataParallel seem ad-hoc
proxy_log_dir = "proxy_log.log"  # FIXME: ad-hoc
