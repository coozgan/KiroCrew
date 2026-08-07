import { Plus, ChevronDown, LayoutGrid } from 'lucide-react'
import { SendBtn } from './ui'
import {
  DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem,
} from './ui/dropdown-menu'
import { i18nT } from '../i18n/t'

/**
 * Split button for creating a job: primary half starts blank, the ▾ half offers
 * the template gallery.
 *
 * They used to be two sibling buttons in the header strip, which read as two
 * unrelated actions — they are one intent ("make a new job") with two starting
 * points, and only one of them is the common path. Same shape as the prerelease
 * FeedbackPill: one control, two hit targets, no separator line needed between
 * them because the halves already differ in width and glyph.
 */
export default function AddJobSplitButton({ onBlank, onBrowseTemplates }: {
  onBlank: () => void
  onBrowseTemplates: () => void
}) {
  return (
    <span className="inline-flex items-stretch overflow-hidden rounded-md">
      <SendBtn className="!rounded-none !border-r-0" onClick={onBlank}>
        <span className="flex items-center gap-1.5">
          <Plus size={14} aria-hidden="true" />
          {i18nT('pages.schedulePage.add_job')}
        </span>
      </SendBtn>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <SendBtn
            className="!rounded-none !px-1.5"
            aria-label={i18nT('pages.schedulePage.browse_schedule_templates')}
            title={i18nT('pages.schedulePage.browse_schedule_templates')}
          >
            <ChevronDown size={14} aria-hidden="true" />
          </SendBtn>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="min-w-[200px]">
          <DropdownMenuItem onSelect={onBrowseTemplates}>
            <LayoutGrid size={13} className="shrink-0 text-accent" />
            <span>{i18nT('pages.schedulePage.browse_all_templates')}</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </span>
  )
}
