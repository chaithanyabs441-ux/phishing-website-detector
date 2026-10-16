# Rollback

From the project root, restore only the three changed application files:

```powershell
Copy-Item patch\backups\pre_threshold_fix_20261012_192700\webapp\app.py webapp\app.py -Force
Copy-Item patch\backups\pre_threshold_fix_20261012_192700\webapp\templates\index.html webapp\templates\index.html -Force
Copy-Item patch\backups\pre_threshold_fix_20261012_192700\webapp\static\style.css webapp\static\style.css -Force
```

The original-file backups are under `patch/backups/pre_threshold_fix_20261012_192700/`. Keep the patch directory until the restored application has been checked.
