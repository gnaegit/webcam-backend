import ids_peak.ids_peak as idsp

def parameter_access_status(status):
    if status==idsp.NodeAccessStatus_NotAvailable:
        return "Not Available"
    elif status==idsp.NodeAccessStatus_ReadOnly:
        return "Read Only"
    elif status==idsp.NodeAccessStatus_WriteOnly:
        return "Write Only"
    elif status==idsp.NodeAccessStatus_ReadWrite:
        return "Read/Write"
    elif status==idsp.NodeAccessStatus_NotImplemented:
        return "Not Implemented"
    else:
        return f"Unknown: {status}"