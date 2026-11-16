; The Cutlass installer for Windows, built by scripts/windows_installer.py.
;
; It installs for the current user, so it needs no administrator: the app
; goes in %LOCALAPPDATA%\Programs\Cutlass, with a Start menu entry, an
; entry in Installed apps, and "Open in Cutlass" for folders (a project)
; and for movies, sound and pictures. /S installs silently; /D= sets the
; folder.
;
; Defines from the script: VERSION, ARCH (x86_64 or aarch64, the processor
; cutlass.exe is built for), SOURCE (the folder holding cutlass.exe and
; cutlass.ico), OUTFILE.

Unicode true
!include LogicLib.nsh
!include x64.nsh
SetCompressor /SOLID lzma
RequestExecutionLevel user

Name "Cutlass"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\Cutlass"
InstallDirRegKey HKCU "Software\Cutlass" "InstallDir"
Icon "${SOURCE}\cutlass.ico"
UninstallIcon "${SOURCE}\cutlass.ico"
VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "Cutlass"
VIAddVersionKey "FileDescription" "Cutlass installer"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "Licensed MIT OR Apache-2.0"

!define UNINSTALL "Software\Microsoft\Windows\CurrentVersion\Uninstall\Cutlass"
!define APPKEY "Software\Classes\Applications\cutlass.exe"

!macro Supported EXT
  WriteRegStr HKCU "${APPKEY}\SupportedTypes" "${EXT}" ""
  WriteRegStr HKCU "Software\Classes\${EXT}\OpenWithList\cutlass.exe" "" ""
!macroend

!macro Unsupported EXT
  DeleteRegKey HKCU "Software\Classes\${EXT}\OpenWithList\cutlass.exe"
!macroend

; An Arm build runs only on Arm Windows. An x86_64 build runs on x64
; Windows, and on Arm Windows through its emulation, but not on 32-bit
; Windows.
Function .onInit
  !if "${ARCH}" == "aarch64"
    ${IfNot} ${IsNativeARM64}
      MessageBox MB_ICONSTOP "This installer is for Windows on Arm. Download Cutlass-x86_64-setup.exe for this PC." /SD IDOK
      Abort
    ${EndIf}
  !else if "${ARCH}" == "x86_64"
    ${IfNot} ${RunningX64}
      MessageBox MB_ICONSTOP "Cutlass needs 64-bit Windows." /SD IDOK
      Abort
    ${EndIf}
  !else
    !error "ARCH is ${ARCH}, not x86_64 or aarch64"
  !endif
FunctionEnd

Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles

Section "Cutlass"
  SetOutPath "$INSTDIR"
  File "${SOURCE}\cutlass.exe"
  File "${SOURCE}\cutlass.ico"
  WriteUninstaller "$INSTDIR\uninstall.exe"

  CreateShortCut "$SMPROGRAMS\Cutlass.lnk" "$INSTDIR\cutlass.exe" "" "$INSTDIR\cutlass.ico"

  WriteRegStr HKCU "Software\Cutlass" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTALL}" "DisplayName" "Cutlass"
  WriteRegStr HKCU "${UNINSTALL}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINSTALL}" "DisplayIcon" "$INSTDIR\cutlass.ico"
  WriteRegStr HKCU "${UNINSTALL}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTALL}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKCU "${UNINSTALL}" "QuietUninstallString" '"$INSTDIR\uninstall.exe" /S'
  WriteRegDWORD HKCU "${UNINSTALL}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTALL}" "NoRepair" 1

  ; "Open with" for media: the app and the kinds it opens.
  WriteRegStr HKCU "${APPKEY}" "FriendlyAppName" "Cutlass"
  WriteRegStr HKCU "${APPKEY}\DefaultIcon" "" "$INSTDIR\cutlass.ico"
  WriteRegStr HKCU "${APPKEY}\shell\open\command" "" '"$INSTDIR\cutlass.exe" "%1"'
  !insertmacro Supported ".mp4"
  !insertmacro Supported ".mov"
  !insertmacro Supported ".m4v"
  !insertmacro Supported ".mkv"
  !insertmacro Supported ".webm"
  !insertmacro Supported ".m4a"
  !insertmacro Supported ".mp3"
  !insertmacro Supported ".wav"
  !insertmacro Supported ".aac"
  !insertmacro Supported ".png"
  !insertmacro Supported ".jpg"
  !insertmacro Supported ".jpeg"
  !insertmacro Supported ".gif"
  !insertmacro Supported ".webp"

  ; A project is a folder: "Open in Cutlass" on any folder.
  WriteRegStr HKCU "Software\Classes\Directory\shell\Cutlass" "" "Open in Cutlass"
  WriteRegStr HKCU "Software\Classes\Directory\shell\Cutlass" "Icon" "$INSTDIR\cutlass.ico"
  WriteRegStr HKCU "Software\Classes\Directory\shell\Cutlass\command" "" '"$INSTDIR\cutlass.exe" "%1"'
SectionEnd

Section "Uninstall"
  Delete "$SMPROGRAMS\Cutlass.lnk"
  Delete "$INSTDIR\cutlass.exe"
  Delete "$INSTDIR\cutlass.ico"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINSTALL}"
  DeleteRegKey HKCU "${APPKEY}"
  !insertmacro Unsupported ".mp4"
  !insertmacro Unsupported ".mov"
  !insertmacro Unsupported ".m4v"
  !insertmacro Unsupported ".mkv"
  !insertmacro Unsupported ".webm"
  !insertmacro Unsupported ".m4a"
  !insertmacro Unsupported ".mp3"
  !insertmacro Unsupported ".wav"
  !insertmacro Unsupported ".aac"
  !insertmacro Unsupported ".png"
  !insertmacro Unsupported ".jpg"
  !insertmacro Unsupported ".jpeg"
  !insertmacro Unsupported ".gif"
  !insertmacro Unsupported ".webp"
  DeleteRegKey HKCU "Software\Classes\Directory\shell\Cutlass"
  DeleteRegKey HKCU "Software\Cutlass"
  ; The user's projects and settings stay.
SectionEnd
