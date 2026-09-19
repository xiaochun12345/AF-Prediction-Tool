"""Run the calculator locally; tolerate a malformed Windows certificate store."""
import argparse
from pathlib import Path
import ssl
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8501)
    args=parser.parse_args()
    try:
        ssl.create_default_context()
    except ssl.SSLError:
        # Use the same maintained public CA bundle used by requests. TLS peer
        # verification and hostname checking remain enabled; no system changes.
        import certifi
        original=ssl.create_default_context
        def context(purpose=ssl.Purpose.SERVER_AUTH,*,cafile=None,capath=None,cadata=None):
            if cafile is None and capath is None and cadata is None:cafile=certifi.where()
            return original(purpose,cafile=cafile,capath=capath,cadata=cadata)
        check=context()
        assert check.verify_mode==ssl.CERT_REQUIRED and check.check_hostname
        ssl.create_default_context=context
        print('Using the certifi CA bundle for this process; TLS verification remains enabled.',flush=True)
    sys.argv=['streamlit','run',str(Path(__file__).with_name('app.py')),
              '--server.address','127.0.0.1','--server.port',str(args.port),
              '--server.headless','true','--browser.gatherUsageStats','false']
    from streamlit.web.cli import main as streamlit_main
    streamlit_main()


if __name__=='__main__':main()
