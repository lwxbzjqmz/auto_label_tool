#激活http://127.0.0.1:5051,加载grounding_dino至缓存
$Path="W:\openmv-ide\object_file\auto_label_tool\backend\large_model\grounding_dino"
$Conda_Env_Name="grounding_dino"

Set-Location -Path $Path

conda activate $Conda_Env_Name

python.exe $Path\worker.py
