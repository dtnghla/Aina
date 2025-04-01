### Input
- The point cloud data is not labeled
- When a new modeling project started, our system will be given the point cloud data and a Revit template. 
- A Revit template is a .rvt file contains but not limited to coordination system, families, levels. A Revit template doesn't have any wall or pipe.
---
### Output
- We must build 2 applications: a Python application to analyze the point cloud and a Revit add-in to create the 3D model from the analysis.
- We build the Python application on Ubuntu 20.04 on WSL2 on Windows 10. Use miniconda3 as environment manager.
- We use Revit 2022 API and .NET 4.8 to build the add-in in Visual Studio 2019 Pro.
---