"""Stage a new image context from verified replay/native outputs, without building."""
from pathlib import Path
import argparse,hashlib,json,shutil
P=Path(__file__).resolve().parents[1]
BASE='vllm/vllm-openai-xpu@sha256:010bdc441c72ee222f459d3b688bb41317525f1bd7223afe6218490b0c526832'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--source',type=Path,required=True);a.add_argument('--native',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    assert not args.output.exists(),'output directory must be new'
    for x in json.loads((P/'manifests/accepted-source.json').read_text()):assert sha(args.source/x['path'])==x['sha256'],x['path']
    root=args.output/'overlay';site=root/'opt/venv/lib/python3.12/site-packages';site.mkdir(parents=True)
    for p in (P/'accepted-overlay').rglob('*'):
        if p.is_file():
            rel=p.relative_to(P/'accepted-overlay');dst=site/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(args.source/rel,dst)
    target=site/'vllm/v1/worker/gpu/spec_decode/dflash2/quality_calibration.pt';shutil.copy2(P/'generated/quality_calibration.pt',target)
    for x in json.loads((P/'manifests/native-builds.json').read_text()):
        src=args.native/x['family']/Path(x['installed_path']).name;assert sha(src)==x['sha256'],str(src)
        dst=root/x['installed_path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    shutil.copytree(P/'support/b70-flashnext',root/'opt/b70-flashnext',dirs_exist_ok=True)
    shutil.copytree(P/'image-support',root,dirs_exist_ok=True)
    image=json.loads((P/'manifests/final-image.json').read_text());config=image['Config']
    lines=['# Reassembly over the pinned vendor binary image; produces a NEW image identity.','FROM '+BASE,'COPY overlay/ /']
    for value in config['Env']:
        k,v=value.split('=',1);lines.append('ENV '+k+'='+json.dumps(v))
    lines+=['WORKDIR '+config['WorkingDir'],'ENTRYPOINT '+json.dumps(config['Entrypoint']),'CMD '+json.dumps(config['Cmd'])]
    (args.output/'Dockerfile').write_text('\n'.join(lines)+'\n')
    print(args.output.resolve())
if __name__=='__main__':main()
