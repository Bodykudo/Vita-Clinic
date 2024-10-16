import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';

import { cn } from '@/lib/utils';

const badgeVariants = cva(
  'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
  {
    variants: {
      variant: {
        default:
          'border-transparent bg-blue-600 text-gray-100 hover:bg-blue-500/90',
        female:
          'border-transparent bg-rose-300 text-rose-800 hover:bg-rose-200',
        active:
          'border-transparent bg-green-800 text-gray-100 hover:bg-green-700',
        inactive:
          'border-transparent bg-gray-300 text-gray-800 hover:bg-gray-200',
        secondary:
          'border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80',
        destructive:
          'border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80',
        completed:
          'border-transparent bg-green-800 text-gray-100 hover:bg-green-700',
        pending:
          'border-transparent bg-orange-500 text-gray-100 hover:bg-orange-400',
        cancelled:
          'border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80',
        approved:
          'border-transparent bg-primary text-destructive-foreground hover:bg-primary/80',
        rejected:
          'border-transparent bg-yellow-500 text-destructive-foreground hover:bg-yellow-500/80',
        outline: 'text-foreground',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
