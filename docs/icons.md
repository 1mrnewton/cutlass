# Icons

UI icons are the free Hugeicons set: `@hugeicons/react` and `@hugeicons/core-free-icons`. Do not install a `@hugeicons-pro` package.

Add an icon by importing it in `src/editor/icons.tsx` and putting it on the `icons` map. Render it with `Icon`. The blade in that file is the only custom mark.

| Name | Free icon | Where |
| --- | --- | --- |
| media | `Image01Icon` | Library rail |
| audio | `AudioWave01Icon` | Library rail, audio tracks |
| text | `TextIcon` | Library rail, text tracks |
| stickers | `StickerIcon` | Library rail, sticker tracks |
| effects | `SparklesIcon` | Library rail, effect tracks |
| transitions | `TransitionLeftIcon` | Library rail |
| filters | `FilterIcon` | Library rail, filter tracks |
| folder | `Folder01Icon` | Project bin folders |
| adjustments | `SlidersHorizontalIcon` | Library rail, adjust tracks |
| templates | `LayoutTemplateIcon` | Library rail |
| video | `Video01Icon` | Video tracks |
| textVideo | `ImageToVideoIcon` | Text-to-video tracks |
| composition | `LayerIcon` | Composition tracks |
| plugin | `MagicWand01Icon` | Plugin-effect tracks |
| ad | `CubeIcon` | Ad tracks |
| mixed | `BlendIcon` | Mixed tracks |
| play | `PlayIcon` | Transport |
| plus | `PlusSignIcon` | Library card, add to the timeline |
| pause | `PauseIcon` | Transport |
| camera | `Camera01Icon` | Transport, still frame |
| chevron | `ArrowRight01Icon` | Clip menu submenu, inspector sections |
| chevronLeft | `ArrowLeft01Icon` | Inspector keyframe step |
| alignLeft | `AlignLeftIcon` | Video inspector |
| alignCenter | `AlignHorizontalCenterIcon` | Video inspector |
| alignRight | `AlignRightIcon` | Video inspector |
| alignTop | `AlignTopIcon` | Video inspector |
| alignMiddle | `AlignVerticalCenterIcon` | Video inspector |
| alignBottom | `AlignBottomIcon` | Video inspector |
| keyframe | `KeyframeIcon` | Video inspector |
| reset | `RefreshIcon` | Video inspector |
| rotate | `RotateClockwiseIcon` | Video inspector, timeline toolbar |
| search | `Search01Icon` | Animation search |
| star | `StarIcon` | Text to speech, favorite a voice |
| bold | `TextBoldIcon` | Text inspector |
| italic | `TextItalicIcon` | Text inspector |
| underline | `TextUnderlineIcon` | Text inspector |
| textLeft | `TextAlignLeftIcon` | Text inspector |
| textCenter | `TextAlignCenterIcon` | Text inspector |
| textRight | `TextAlignRightIcon` | Text inspector |
| justifyLeft | `TextAlignJustifyLeftIcon` | Text inspector |
| justifyCenter | `TextAlignJustifyCenterIcon` | Text inspector |
| justifyRight | `TextAlignJustifyRightIcon` | Text inspector |
| unavailable | `UnavailableIcon` | Animation none |
| undo | `Undo03Icon` | Timeline toolbar |
| redo | `Redo03Icon` | Timeline toolbar |
| select | `Cursor02Icon` | Timeline toolbar |
| split | `SquareSplitHorizontalIcon` | Timeline toolbar |
| delete | `Delete02Icon` | Timeline toolbar |
| freeze | `SnowflakeIcon` | Timeline toolbar |
| reverse | `RewindIcon` | Timeline toolbar |
| mirror | `FlipHorizontalIcon` | Timeline toolbar |
| crop | `CropIcon` | Timeline toolbar |
| transcribe | `SpeechToTextIcon` | Timeline toolbar |
| splitScenes | `ScissorRectangleIcon` | Timeline toolbar, split scenes |
| snap | `AlignSelectionIcon` | Timeline toolbar |
| magnet | `MagnetIcon` | Timeline toolbar, main-track magnet |
| linkage | `Link01Icon` | Timeline toolbar |
| marker | `Flag01Icon` | Timeline toolbar |
| beat | `MusicNote01Icon` | Timeline toolbar |
| zoomIn | `ZoomInAreaIcon` | Timeline toolbar |
| zoomOut | `ZoomOutAreaIcon` | Timeline toolbar |
| zoomFit | `FitToScreenIcon` | Timeline toolbar, zoom to fit |
| guides | `RulerIcon` | Timeline toolbar, show or hide reference lines |
| lock | `SquareLock01Icon` | Timeline toolbar and track labels |

Stroke weight is `1.5`. Color is `currentColor`, so a button's text color paints the icon.
