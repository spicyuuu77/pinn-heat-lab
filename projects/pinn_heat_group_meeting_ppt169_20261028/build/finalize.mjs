import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const project = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const workspace = path.resolve(project,'../..');
const skill = 'C:/Users/Kendrick何/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
process.env.RUNTIME_NODE_MODULES='C:/Users/Kendrick何/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {finalizePresentation} = await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const out=path.join(project,'exports','一维热传导PINN复现_组会汇报.pptx');
await fs.mkdir(path.dirname(out),{recursive:true});
const result=await finalizePresentation({
  workspaceDir:workspace,
  candidatePath:path.join(project,'build','candidate.pptx'),
  finalPath:out,
  pythonExecutable:'C:/Users/Kendrick何/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
  layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
  layoutArgs:['--expected-slide-size-emu','9144000,5184775','--validate-bullet-geometry','--validate-heading-fit'],
  explicitTotalSlideCount:5,
  requiredNativeChartOwnerSlides:[4],
  requiredNativeTableOwnerSlides:[],
  materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'design',families:['DengXian','Cambria Math']},
  verifyArtifactToolImport:true,
  receiptPath:path.join(project,'build','validation.json')
});
console.log(JSON.stringify(result));
